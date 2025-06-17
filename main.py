from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

app = FastAPI()

@app.post("/process")
async def process(file: UploadFile = File(...)):
    try:
        # קריאת הקובץ
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))

        # ניתוח (דוגמה בסיסית)
        summary = df.describe().to_string()

        # גרף לדוגמה
        plt.figure(figsize=(6, 4))
        sns.histplot(df[df.columns[0]])
        buf = BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        graph_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return JSONResponse({
            "summary": summary,
            "graph": graph_base64
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
