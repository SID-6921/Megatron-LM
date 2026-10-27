# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Coverage for occurrence-based FSDP prefetch and weak module ownership."""

import gc
from unittest.mock import MagicMock
from weakref import ref

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    Placements,
    PrefetchOrder,
    fully_shard,
    fully_shard_context,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.module import FsdpModule
from tests.unit_tests.distributed.mfsdp_v2.test_annotation import NvtxEvent, _setup_nvtx_recording


def test_recorded_prefetch_follows_execution(distributed_setup, monkeypatch):
    """Prefetch follows execution when submodules are registered in the opposite order."""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.second = nn.Linear(4, 4, bias=False)
            self.first = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.second(self.first(x))

    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    model = Model().to(device)
    placements = Placements(
        dp_axes=[0], parameter=[Shard(0)], gradient=[Shard(0)], optimizer=[Shard(0)]
    )
    with fully_shard_context(
        FsdpContext(device=device, prefetch_order=PrefetchOrder.RECORDED)
    ) as context:
        for layer in model.children():
            fully_shard(layer, mesh, placements)
        fully_shard(model, mesh, placements)

    inputs = torch.ones(2, 4, device=device)
    model(inputs).sum().backward()
    assert list(context.forward_order) == [model, model.first, model.second]
    assert list(context.backward_order) == [model, model.second, model.first]

    events: list[NvtxEvent] = []
    _setup_nvtx_recording(monkeypatch, events)
    model(inputs).sum().backward()

    assert [
        event
        for event in events
        if event.kind == "push" and event.operation in ("forward", "backward", "prefetch")
    ] == [
        NvtxEvent("push", "<root>", "forward"),
        NvtxEvent("push", "first", "prefetch"),
        NvtxEvent("push", "first", "forward"),
        NvtxEvent("push", "second", "prefetch"),
        NvtxEvent("push", "second", "forward"),
        NvtxEvent("push", "<root>", "backward"),
        NvtxEvent("push", "second", "prefetch"),
        NvtxEvent("push", "second", "backward"),
        NvtxEvent("push", "first", "prefetch"),
        NvtxEvent("push", "first", "backward"),
    ]


@pytest.mark.parametrize(
    'budget, expected',
    [
        (None, [[1], [0], [0], [2], []]),
        (0, [[], [], [], [], []]),
        (2, [[1, 0], [0, 0], [0, 2], [2], []]),
    ],
)
def test_prefetch_replays_occurrences(budget, expected):
    """Reused modules have occurrence-specific successors under every prefetch budget."""
    modules = [MagicMock(spec=FsdpModule, num_parameter_elements=1) for _ in range(3)]
    prefetched = []
    for module in modules:
        # Prefetch passes the successor's compute phase; record only which module it targets.
        module._unshard_parameter_groups.side_effect = (
            lambda _phase, target=module: prefetched.append(target)
        )
    sequence = [modules[index] for index in (0, 1, 0, 0, 2)]
    order = ExecutionOrder()
    for module in sequence:
        order.append(module)
    # A second cycle checks cursor reset; each lookahead must leave demand calls intact.
    for _ in range(2):
        for module, targets, successor in zip(sequence, expected, sequence[1:] + [None]):
            prefetched.clear()
            FsdpModule._prefetch_parameter_groups(module, order, budget)
            assert prefetched == [modules[index] for index in targets]
            assert order.next_item() is successor


def test_unique_calls_resynchronize_and_repeated_calls_validate():
    """Unique VPP chunk calls select their position; ambiguous repeats stay strict."""
    chunk0, layer0, chunk1, layer1 = (nn.Module() for _ in range(4))
    order = ExecutionOrder([chunk0, layer0, chunk1, layer1])
    # Grouped VPP repeats chunk 0 before switching to chunk 1, in one shared order.
    for chunk in (chunk0, chunk0, chunk0, chunk1, chunk1, chunk0):
        layer = layer0 if chunk is chunk0 else layer1
        order.advance(chunk)
        assert order.next_item() is layer
        order.advance(layer)
        assert order.next_item() is (chunk1 if chunk is chunk0 else None)

    order = ExecutionOrder([chunk0, layer0, chunk0])
    order.advance(chunk0)
    assert order.next_item() is layer0
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(chunk0)
    # A rejected repeated call must not consume the expected occurrence.
    order.advance(layer0)
    assert order.next_item() is chunk0
    order.advance(chunk0)
    assert order.next_item() is None
    # Unique calls remain valid anchors even when the sequence also has repeats.
    order.advance(layer0)
    order.advance(chunk0)
    assert order.next_item() is None

    empty = ExecutionOrder()
    assert empty.next_item() is None
    with pytest.raises(RuntimeError, match='diverged'):
        empty.advance(chunk0)


def test_iteration_and_weak_references():
    """Traversal preserves repeated occurrences without retaining deleted modules."""
    first, second = nn.Module(), nn.Module()
    order = ExecutionOrder([first, second, first, second])
    assert list(order) == [first, second, first, second]

    second_ref = ref(second)
    del second
    gc.collect()
    assert second_ref() is None
    assert list(order) == [first, first]
    order.advance(first)
    assert order.next_item() is None
    # Traversal skips dead references, but the demand cursor must not skip a call.
    with pytest.raises(RuntimeError, match='diverged'):
        order.advance(first)
