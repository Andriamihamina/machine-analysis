import json


def fake_datas() -> str:
    datas = [
        {"timestamp": "2025-05-11T07:08:49.154000", "energy_value": 25.96},
        {
            "timestamp": "2025-05-11T07:08:49.413000",
            "energy_value": -300,
        },
        {"timestamp": "2025-05-11T07:08:49.672000"},
        {"energy_value": 25.83},
        {"timestamp": "2025-05-11T07:08:49.931000", "energy_value": "27.5"},
    ]
    return json.dumps(datas)


def datas_need_renaming() -> str:
    datas = [
        {"time": "2025-05-11T07:08:49.154000", "value": 25.96},
        {
            "time": "2025-05-11T07:08:49.413000",
            "value": -300,
        },
        {"time": "2025-05-11T07:08:49.672000"},
        {"value": 25.83},
        {"time": "2025-05-11T07:08:49.931000", "value": "27.5"},
    ]
    return json.dumps(datas)
