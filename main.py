from services.extract_service.app.main import SETTINGS, app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port)
