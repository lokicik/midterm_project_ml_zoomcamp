import json
import requests
host = ""
url = f'http://{host}/predict'

patient = {
    "sex": 0.0,
    "age": 0.3076923077,
    "height": 0.0909090909,
    "weight": 0.15,
    "waistline": 0.2235294118,
    "sight_left": 0.4210526316,
    "sight_right": 0.5789473684,
    "hear_left": 1.0,
    "hear_right": 0.0,
    "SBP": 0.2928571429,
    "DBP": 0.214953271,
    "BLDS": 0.0634920635,
    "tot_chole": 0.1743295019,
    "HDL_chole": 0.3655172414,
    "LDL_chole": 0.1710794297,
    "triglyceride": 0.0454296661,
    "hemoglobin": 0.1352941176,
    "urine_protein": 0.0,
    "serum_creatinine": 0.0491803279,
    "SGOT_AST": 0.0200601805,
    "SGOT_ALT": 0.0103806228,
    "gamma_GTP": 0.0122767857,
    "BMI": 0.2651034483,
    "BMI_Category": 0.3333333333,
    "MAP": 0.2561728395,
    "Liver_Enzyme_Ratio": 0.1258141412,
    "Anemia_Indicator": 1.0
}

response = requests.post(url, json=patient).json()
pretty_response = json.dumps(response, indent=4)

print(pretty_response)

