from fastapi import FastAPI, File, UploadFile
import torch
import translation.IndicTrans2.NeMo.nemo.collections.asr as nemo_asr
import os
import subprocess
from translation.IndicTrans2.inference.engine import Model
from pydantic import BaseModel

class obj(BaseModel):
    text: str
    source: str

# Initialize FastAPI app
app = FastAPI()

model_en = Model('./translation/models/indic-en/fairseq_model', model_type="fairseq")

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path='translation/IndicTrans2/NeMo/hi.nemo')
model.freeze()  # Set the model in inference mode
model = model.to(device)  # Move the model to the correct device

@app.post("/translate/")
async def translate(obj:obj):
    text = obj.text
    source = obj.source
    translation = model_en.translate_paragraph(text, source, 'eng_Latn')
    return {"translation": translation}

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    # Save the uploaded file to a temporary location
    input_filename = "input_audio.wav"
    output_filename = "sample_audio_infer_ready.wav"
    
    with open(input_filename, "wb") as temp_file:
        temp_file.write(await file.read())
    
    # Convert the audio file using ffmpeg (to mono, 16kHz)
    try:
        # Execute ffmpeg command to convert the audio file
        command = [
            "ffmpeg", "-i", input_filename, "-ac", "1", "-ar", "16000", output_filename, "-y"
        ]
        subprocess.run(command, check=True)
    except Exception as e:
        return {"error": f"FFmpeg failed: {e}"}

    try:
        # Perform transcription using the model
        model.cur_decoder = "ctc"
        transcription = model.transcribe([output_filename], batch_size=1, logprobs=False, language_id='hi')[0]

        # Clean up the temporary files
        os.remove(input_filename)
        os.remove(output_filename)
        
        return {"transcription": transcription}
    
    except Exception as e:
        return {"error": str(e)}

# For local testing, run using `uvicorn main:app --reload`
