"""uvicorn 엔트리포인트.

실행
----
    uvicorn solar_thermal.api.main:app --host 0.0.0.0 --port 8080
"""

from .app import create_app

app = create_app()
