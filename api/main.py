import shutil
import os
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO

from model.pipeline import run_pipeline_and_get_summary

app = FastAPI(title="Fall Detection API")

# Le modèle est chargé UNE SEULE FOIS au démarrage de l'API,
# pas à chaque requête (ce serait très lent sinon)
model = YOLO("yolo11n-pose.pt")


@app.post("/analyze")
async def analyze_video(file: UploadFile):
    """
    Reçoit une vidéo, lance le pipeline complet dessus,
    retourne un résumé structuré en JSON.
    """
    # 1. Sauvegarder temporairement la vidéo reçue
    temp_input_path = f"temp_{file.filename}"
    with open(temp_input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Définir où sauvegarder la vidéo annotée en sortie
    temp_output_path = f"output_{file.filename}"

    # 3. Lancer le pipeline complet
    summary = run_pipeline_and_get_summary(
        temp_input_path, temp_output_path, model
    )

    # 4. Nettoyer les fichiers temporaires
    os.remove(temp_input_path)
    os.remove(temp_output_path)

    # 5. Retourner le résultat -- FastAPI convertit automatiquement
    #    ce dictionnaire Python en JSON
    return {
        "video_processed": True,
        "filename": file.filename,
        **summary
    }


@app.get("/")
def health_check():
    """Un endpoint simple pour vérifier que l'API tourne."""
    return {"status": "Fall Detection API is running"}