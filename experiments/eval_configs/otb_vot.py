eval_cfgs = [
    dict(
        metrics=dict(type='OPE'),
        dataset=dict(type='OTB100'),
        hypers=dict(
            epoch=list(range(31, 51, 2)),
            window=dict(
                weight=[0.200, 0.300, 0.400]
            )
        )
    ),
    dict(
        metrics=dict(
            type='Restart',
            skipping=5,
            low=100,
            high=356,
            peak=160,
        ),
        dataset=dict(type='VOT17'),
        hypers=dict(
            epoch=list(range(31, 51, 2)),
            window=dict(
                weight=[0.200, 0.250, 0.300, 0.350, 0.400]
            )
        )
    ),
]
