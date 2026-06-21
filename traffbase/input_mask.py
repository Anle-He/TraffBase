from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class InputMaskSettings:
    '''Resolved settings for test-time input masking.'''

    steps: int
    repeats: int
    description: str


def resolve_input_mask_settings(
    mask_config: dict[str, Any], sequence_length: int
) -> InputMaskSettings | None:
    '''Validate test mask configuration and resolve the number of masked steps.'''

    if not mask_config.get('enabled', False):
        return None

    if sequence_length <= 0:
        raise ValueError('sequence_length must be greater than 0')

    ratio = mask_config.get('ratio')
    steps = mask_config.get('steps')

    if (ratio is None) == (steps is None):
        raise ValueError(
            'Exactly one of TEST.input_mask.ratio or TEST.input_mask.steps must be set'
        )

    repeats = mask_config.get('repeats', 5)
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError('TEST.input_mask.repeats must be a positive integer')

    if ratio is not None:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError('TEST.input_mask.ratio must be a number')
        if not 0 < ratio <= 1:
            raise ValueError('TEST.input_mask.ratio must be in the range (0, 1]')

        masked_steps = int(ratio * sequence_length + 0.5)
        if masked_steps == 0:
            raise ValueError(
                'TEST.input_mask.ratio is too small to mask any input step'
            )
        description = f'ratio={ratio:g}, steps={masked_steps}'
    else:
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise ValueError('TEST.input_mask.steps must be an integer')
        if not 1 <= steps <= sequence_length:
            raise ValueError(
                'TEST.input_mask.steps must be between 1 and the input sequence length'
            )

        masked_steps = steps
        description = f'steps={masked_steps}'

    return InputMaskSettings(
        steps=masked_steps,
        repeats=repeats,
        description=description,
    )


def apply_random_time_mask(
    inputs: torch.Tensor, steps: int, seed: int
) -> torch.Tensor:
    '''Mask shared time steps across all nodes for each sample.'''

    if inputs.ndim < 3:
        raise ValueError('inputs must have shape [batch, time, ...]')

    batch_size, sequence_length = inputs.shape[:2]
    if not 1 <= steps <= sequence_length:
        raise ValueError('steps must be between 1 and the input sequence length')

    time_mask = torch.zeros(
        batch_size,
        sequence_length,
        dtype=torch.bool,
        device='cpu',
    )
    generator = torch.Generator(device='cpu')
    for sample_index in range(batch_size):
        generator.manual_seed(seed + sample_index)
        masked_indices = torch.randperm(
            sequence_length,
            generator=generator,
            device='cpu',
        )[:steps]
        time_mask[sample_index, masked_indices] = True

    masked_inputs = inputs.clone()
    broadcast_shape = (batch_size, sequence_length) + (1,) * (inputs.ndim - 2)
    expanded_mask = time_mask.reshape(broadcast_shape).to(inputs.device)
    masked_inputs.masked_fill_(expanded_mask, 0)

    return masked_inputs
