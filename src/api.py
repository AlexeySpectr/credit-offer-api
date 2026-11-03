import joblib
import pandas as pd
from datetime import date
from fastapi import FastAPI,Depends
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse
from src.features import build_features
from contextlib import asynccontextmanager
from database import Base, engine,get_db
from sqlalchemy.orm import Session
from db_models import PredictionRecord
from sqlalchemy import select
from pydantic import ConfigDict
import logging
import time
from prometheus_fastapi_instrumentator import Instrumentator


from fastapi import Request

from logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)






model = joblib.load('models/model.joblib')

@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="Credit Offer Prediction API",
    version="1.0.0",
    lifespan=lifespan
)
Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics"
)

@app.middleware("http")
async def log_requests(request: Request, call_next):

    start_time = time.perf_counter()

    method = request.method
    path = request.url.path

    logger.info(
        "Request started: %s %s",
        method,
        path
    )

    try:
        response = await call_next(request)

        duration = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Request completed: %s %s status=%s duration=%.2fms",
            method,
            path,
            response.status_code,
            duration
        )

        return response

    except Exception:
        duration = (time.perf_counter() - start_time) * 1000

        logger.exception(
            "Request failed: %s %s duration=%.2fms",
            method,
            path,
            duration
        )

        raise

class CreditApplication(BaseModel):
    decision_day: str

    loan_amount_last: float | None = None
    overdraft_limit_min: float | None = None
    overdraft_limit_max: float | None = None

    offered_rate: float
    cb_rate: float

    fl_adminarea: str | None = None


class PredictionResponse(BaseModel):
    prediction: int
    probability: float

class PredictionHistoryResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    prediction: int
    probability: float
    offered_rate: float
    cb_rate: float

@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict', response_model=PredictionResponse)
def predict(
    application: CreditApplication,
    db: Session = Depends(get_db)
):
    df = pd.DataFrame([application.model_dump()])

  
    X = build_features(df)

    logger.info("ML model prediction started")

    probability = float(model.predict_proba(X)[0, 1])

    prediction = int(probability >= 0.5)

    logger.info(
        "ML prediction completed: prediction=%s probability=%.4f",
        prediction,
        probability
    )
    record = PredictionRecord(
        decision_day=date.fromisoformat(application.decision_day),
        loan_amount_last=application.loan_amount_last,
        offered_rate=application.offered_rate,
        cb_rate=application.cb_rate,
        fl_adminarea=application.fl_adminarea,
        prediction=prediction,
        probability=probability
    )

    # Сохраняем запись
    db.add(record)
    db.commit()

    return PredictionResponse(
        prediction=prediction,
        probability=probability
    )
@app.get(
    '/history',
    response_model=list[PredictionHistoryResponse]
)
def get_history(db: Session = Depends(get_db)):

    # Формируем SQL-запрос
    statement = (
        select(PredictionRecord)
        .order_by(PredictionRecord.id.desc())
    )

    # Выполняем запрос
    result = db.execute(statement)

    # Получаем ORM-объекты
    records = result.scalars().all()

    return records

@app.get("/form", response_class=HTMLResponse)
def prediction_form():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">

        <title>Credit Offer Prediction</title>

        <style>
            * {
                box-sizing: border-box;
            }

            body {
                font-family: Arial, sans-serif;
                background: #f3f5f9;
                margin: 0;
                padding: 30px 15px;
                color: #243047;
            }

            .container {
                max-width: 750px;
                margin: auto;
                background: white;
                padding: 30px;
                border-radius: 14px;
                box-shadow: 0 5px 25px rgba(0, 0, 0, 0.08);
            }

            h1 {
                text-align: center;
                margin-top: 0;
            }

            .description {
                text-align: center;
                color: #687386;
                margin-bottom: 25px;
            }

            label {
                display: block;
                margin-top: 17px;
                margin-bottom: 7px;
                font-weight: bold;
            }

            input {
                width: 100%;
                padding: 12px;
                border: 1px solid #cbd2df;
                border-radius: 7px;
                font-size: 16px;
            }

            input:focus {
                outline: 2px solid #8ab4f8;
                border-color: #4285f4;
            }

            button {
                width: 100%;
                margin-top: 25px;
                padding: 14px;
                border: none;
                border-radius: 7px;
                background: #2563eb;
                color: white;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
            }

            button:hover {
                background: #1d4ed8;
            }

            button:disabled {
                background: #8ba8e8;
                cursor: wait;
            }

            #result {
                display: none;
                margin-top: 25px;
                padding: 20px;
                border-radius: 8px;
                background: #eff6ff;
                line-height: 1.8;
                overflow-wrap: anywhere;
            }

            #error {
                color: #b91c1c;
                margin-top: 15px;
                overflow-wrap: anywhere;
            }

            .hint {
                font-size: 13px;
                color: #687386;
                margin-top: 5px;
            }

            .result-card {
                background: white;
                padding: 24px;
                border-radius: 12px;
                border: 1px solid #dce3ef;
                text-align: center;
            }

            .result-card h2 {
                margin-top: 0;
                color: #243047;
            }

            .status {
                display: inline-block;
                padding: 10px 22px;
                border-radius: 20px;
                font-weight: bold;
                margin: 10px 0 20px;
            }

            .approved {
                background: #dcfce7;
                color: #166534;
            }

            .rejected {
                background: #fee2e2;
                color: #991b1b;
            }

            .probability-title {
                color: #687386;
                margin-bottom: 5px;
            }

            .probability-value {
                font-size: 36px;
                font-weight: bold;
                color: #2563eb;
            }

            .progress {
                width: 100%;
                height: 12px;
                margin-top: 15px;
                background: #e5e7eb;
                border-radius: 10px;
                overflow: hidden;
            }

            .progress-bar {
                height: 100%;
                background: #2563eb;
                border-radius: 10px;
                transition: width 0.5s ease;
            }

            .explanation {
                margin-top: 20px;
                color: #687386;
            }

            /* История предсказаний */

            .history-section {
                margin-top: 35px;
                border-top: 1px solid #dce3ef;
                padding-top: 25px;
            }

            .history-section h2 {
                text-align: center;
            }

            .history-table-wrapper {
                overflow-x: auto;
                margin-top: 15px;
            }

            table {
                width: 100%;
                border-collapse: collapse;
                font-size: 14px;
            }

            th, td {
                padding: 12px 10px;
                border-bottom: 1px solid #e5e7eb;
                text-align: center;
                white-space: nowrap;
            }

            th {
                background: #eff6ff;
            }

            .history-button {
                margin-top: 12px;
            }

            .history-message {
                margin-top: 12px;
                color: #687386;
                text-align: center;
            }

            @media (max-width: 500px) {
                .container {
                    padding: 20px 15px;
                }

                .probability-value {
                    font-size: 30px;
                }
            }
        </style>
    </head>

    <body>

        <div class="container">

            <h1>Credit Offer Prediction</h1>

            <p class="description">
                Введите параметры заявки для получения прогноза ML-модели.
            </p>

            <!-- Форма предсказания -->

            <form id="prediction-form">

                <label for="decision_day">Дата заявки *</label>
                <input
                    type="date"
                    id="decision_day"
                    required
                >

                <label for="offered_rate">Предложенная ставка (%) *</label>
                <input
                    type="number"
                    id="offered_rate"
                    step="any"
                    required
                    placeholder="Например, 12.5"
                >

                <label for="cb_rate">Ставка ЦБ (%) *</label>
                <input
                    type="number"
                    id="cb_rate"
                    step="any"
                    required
                    placeholder="Например, 8.5"
                >

                <label for="loan_amount_last">Сумма кредита</label>
                <input
                    type="number"
                    id="loan_amount_last"
                    step="any"
                    placeholder="Например, 100000"
                >

                <label for="overdraft_limit_min">
                    Минимальный лимит овердрафта
                </label>

                <input
                    type="number"
                    id="overdraft_limit_min"
                    step="any"
                    placeholder="Можно оставить пустым"
                >

                <label for="overdraft_limit_max">
                    Максимальный лимит овердрафта
                </label>

                <input
                    type="number"
                    id="overdraft_limit_max"
                    step="any"
                    placeholder="Можно оставить пустым"
                >

                <label for="fl_adminarea">Регион</label>

                <input
                    type="text"
                    id="fl_adminarea"
                    placeholder="Введите название региона"
                >

                <p class="hint">
                    Указывайте регион в том формате, который использовался
                    при обучении модели.
                </p>

                <button type="submit" id="submit-button">
                    Получить прогноз
                </button>

            </form>

            <div id="error" role="alert"></div>

            <div id="result" aria-live="polite"></div>


            <!-- История предсказаний -->

            <div class="history-section">

                <h2>История предсказаний</h2>

                <button
                    type="button"
                    id="history-button"
                    class="history-button"
                >
                    Показать историю
                </button>

                <div
                    id="history-message"
                    class="history-message"
                ></div>

                <div class="history-table-wrapper">

                    <table id="history-table" style="display: none;">

                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Prediction</th>
                                <th>Probability</th>
                                <th>Offered rate</th>
                                <th>CB rate</th>
                            </tr>
                        </thead>

                        <tbody id="history-body"></tbody>

                    </table>

                </div>

            </div>

        </div>


        <script>

            // Элементы формы

            const form = document.getElementById("prediction-form");
            const result = document.getElementById("result");
            const error = document.getElementById("error");
            const button = document.getElementById("submit-button");


            // Получение числовых значений

            function getNumber(id) {

                const value = document.getElementById(id).value.trim();

                if (value === "") {
                    return null;
                }

                const number = Number(value);

                if (!Number.isFinite(number)) {
                    throw new Error("Проверьте числовое поле: " + id);
                }

                return number;
            }


            // Отправка предсказания

            form.addEventListener("submit", async (event) => {

                event.preventDefault();

                error.textContent = "";
                result.style.display = "none";

                button.disabled = true;
                button.textContent = "Выполняется прогноз...";

                try {

                    const payload = {

                        decision_day:
                            document.getElementById("decision_day").value,

                        offered_rate: getNumber("offered_rate"),

                        cb_rate: getNumber("cb_rate"),

                        loan_amount_last: getNumber("loan_amount_last"),

                        overdraft_limit_min:
                            getNumber("overdraft_limit_min"),

                        overdraft_limit_max:
                            getNumber("overdraft_limit_max"),

                        fl_adminarea:
                            document.getElementById("fl_adminarea").value.trim()
                            || null
                    };


                    const response = await fetch("/predict", {

                        method: "POST",

                        headers: {
                            "Content-Type": "application/json"
                        },

                        body: JSON.stringify(payload)

                    });


                    const data = await response.json();

                    if (!response.ok) {

                        throw new Error(
                            data.detail
                                ? JSON.stringify(data.detail)
                                : "Ошибка API: " + response.status
                        );

                    }


                    const probability = Number(data.probability);

                    result.replaceChildren();


                    const card = document.createElement("div");

                    card.className = "result-card";


                    const heading = document.createElement("h2");

                    heading.textContent = "Результат прогнозирования";

                    card.appendChild(heading);


                    const status = document.createElement("div");

                    status.className = data.prediction === 1
                        ? "status approved"
                        : "status rejected";

                    status.textContent = data.prediction === 1
                        ? "Класс 1"
                        : "Класс 0";

                    card.appendChild(status);


                    const probabilityTitle = document.createElement("p");

                    probabilityTitle.className = "probability-title";

                    probabilityTitle.textContent = "Вероятность класса 1";

                    card.appendChild(probabilityTitle);


                    const probabilityValue = document.createElement("div");

                    probabilityValue.className = "probability-value";

                    probabilityValue.textContent =
                        (probability * 100).toFixed(2) + "%";

                    card.appendChild(probabilityValue);


                    const progress = document.createElement("div");

                    progress.className = "progress";


                    const progressBar = document.createElement("div");

                    progressBar.className = "progress-bar";

                    progressBar.style.width =
                        Math.max(0, Math.min(100, probability * 100)) + "%";


                    progress.appendChild(progressBar);

                    card.appendChild(progress);


                    const explanation = document.createElement("p");

                    explanation.className = "explanation";

                    explanation.textContent = data.prediction === 1
                        ? "Модель отнесла заявку к классу 1."
                        : "Модель отнесла заявку к классу 0.";

                    card.appendChild(explanation);


                    result.appendChild(card);

                    result.style.display = "block";

                    // Обновляем историю после успешного предсказания
                    await loadHistory();

                } catch (err) {

                    error.textContent =
                        "Не удалось получить прогноз: " + err.message;

                } finally {

                    button.disabled = false;

                    button.textContent = "Получить прогноз";

                }

            });


            // Элементы истории

            const historyButton =
                document.getElementById("history-button");

            const historyBody =
                document.getElementById("history-body");

            const historyTable =
                document.getElementById("history-table");

            const historyMessage =
                document.getElementById("history-message");


            // Получение истории из PostgreSQL через API

            async function loadHistory() {

                historyButton.disabled = true;

                historyButton.textContent = "Загрузка...";

                historyMessage.textContent = "";

                historyBody.replaceChildren();

                historyTable.style.display = "none";


                try {

                    const response = await fetch("/history");

                    if (!response.ok) {

                        throw new Error(
                            "Ошибка API: " + response.status
                        );

                    }


                    const records = await response.json();


                    if (records.length === 0) {

                        historyMessage.textContent =
                            "История пока пуста.";

                        return;

                    }


                    records.forEach(record => {

                        const row = document.createElement("tr");


                        const values = [

                            record.id,

                            record.prediction,

                            (Number(record.probability) * 100).toFixed(2) + "%",

                            record.offered_rate,

                            record.cb_rate

                        ];


                        values.forEach(value => {

                            const cell = document.createElement("td");

                            cell.textContent = value;

                            row.appendChild(cell);

                        });


                        historyBody.appendChild(row);

                    });


                    historyTable.style.display = "table";

                    historyMessage.textContent =
                        "Всего записей: " + records.length;


                } catch (err) {

                    historyMessage.textContent =
                        "Не удалось загрузить историю: " + err.message;

                } finally {

                    historyButton.disabled = false;

                    historyButton.textContent = "Обновить историю";

                }

            }


            // Кнопка загрузки истории

            historyButton.addEventListener("click", loadHistory);

        </script>

    </body>
    </html>
    """

