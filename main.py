from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from bn2en import bn_en_translate as bn2en_translate, en_bn_translate as en2bn_translate


class InputSerializer(BaseModel):
    text: str


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(name="index.html", context={"request": request})


@app.post("/translate/en")
async def translate_en(payload: InputSerializer):
    en = bn2en_translate(payload.text)
    return InputSerializer(text=en)


@app.post("/translate/bn")
async def translate_en(payload: InputSerializer):
    bn = en2bn_translate(payload.text)
    return InputSerializer(text=bn)
