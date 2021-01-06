import requests
from datetime import date, datetime

today = datetime.now()
url = 'http://34.101.71.138:8080/api/v1/user'

def check_in(id):
    try:
        student_info = requests.get(f"{url}?id={id}")
        student_info = student_info.json()
        if len(student_info) > 0:
            student_info = student_info[0]
            checkin = {
                "isLate": 0,
                "time": str(today)
            }
            checkInTimes = student_info['checkInTimes']
            checkInTimes.append(checkin)
            student_info['checkInTimes'] = checkInTimes
            res = requests.put(url, json=student_info)
            return res
    except Exception as e:
        print(str(e))
        return "Exception"
