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
]