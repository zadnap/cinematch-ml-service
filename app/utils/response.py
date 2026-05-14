def success(data={}, message="", status=200):
    return {
        "success": True,
        "data": data,
        "message": message
    }, status


def error(code, message="", status=400):
    return {
        "success": False,
        "error": {
            "code": code,
            "message": message
        }
    }, status