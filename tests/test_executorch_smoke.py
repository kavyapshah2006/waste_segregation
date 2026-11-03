"""Smoke test: the toolchain can export a model to .pte with XNNPACK and run it."""

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime


class TinyNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


def test_export_and_run_xnnpack() -> None:
    torch.manual_seed(0)
    model = TinyNet().eval()
    example = (torch.randn(2, 8),)

    exported = torch.export.export(model, example)
    program = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    method = Runtime.get().load_program(program.buffer).load_method("forward")
    (output,) = method.execute(list(example))

    torch.testing.assert_close(output, model(*example))
