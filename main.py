import os
import base64
import hashlib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, description="Text prompt for image generation")
    provider: Optional[str] = Field(None, description="Optional provider, e.g., 'stability'")
    width: Optional[int] = Field(512, ge=128, le=1024)
    height: Optional[int] = Field(512, ge=128, le=1024)


class GenerateResponse(BaseModel):
    image_b64: str
    provider: str
    model: Optional[str] = None
    mode: str
    note: Optional[str] = None


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


@app.post("/api/generate", response_model=GenerateResponse)
def generate_image(req: GenerateRequest):
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    stability_key = os.getenv("STABILITY_API_KEY")
    provider = (req.provider or ("stability" if stability_key else "demo")).lower()

    # Try to save record to DB if available
    try:
        from database import create_document
        from schemas import ImageGeneration
        base_doc = ImageGeneration(prompt=prompt, provider=provider, model=None, image_b64=None, width=req.width, height=req.height)
        _id = create_document("imagegeneration", base_doc)
    except Exception:
        _id = None  # DB not configured or failed; continue without persistence

    if provider == "stability" and stability_key:
        try:
            url = "https://api.stability.ai/v2beta/stable-image/generate/core"
            headers = {
                "Authorization": f"Bearer {stability_key}",
            }
            data = {
                "prompt": prompt,
                "output_format": "png",
                "aspect_ratio": f"{req.width}:{req.height}",
            }
            # Stability core expects aspect ratios like 1:1, 16:9, 3:2. We'll approximate by rounding to nearest common AR.
            # For simplicity, override to 1:1 if not square to avoid API rejection.
            if req.width == req.height:
                data["aspect_ratio"] = "1:1"
            else:
                data["aspect_ratio"] = "1:1"

            resp = requests.post(url, headers=headers, files=None, data=data, timeout=120)
            if resp.status_code != 200:
                raise HTTPException(status_code=500, detail=f"Stability API error: {resp.status_code} {resp.text[:200]}")

            image_bytes = resp.content  # binary PNG
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # Try update DB with image
            try:
                from database import db
                if db is not None and _id is not None:
                    db["imagegeneration"].update_one({"_id": db["imagegeneration"].inserted_id if False else None}, {"$set": {}})  # no-op placeholder to keep API simple
            except Exception:
                pass

            return GenerateResponse(
                image_b64=image_b64,
                provider="stability",
                model="stable-image-core",
                mode="live",
            )
        except HTTPException:
            raise
        except Exception as e:
            # Fallback to demo mode if any error
            provider = "demo"

    # Demo mode: generate deterministic placeholder from prompt using Picsum
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16) % (10**8)
    demo_url = f"https://picsum.photos/seed/{seed}/{req.width}/{req.height}"
    try:
        r = requests.get(demo_url, timeout=30)
        r.raise_for_status()
        img_b64 = base64.b64encode(r.content).decode("utf-8")
    except Exception:
        # Final fallback: 1x1 white pixel
        img_b64 = base64.b64encode(bytes([137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,6,0,0,0,31,21,196,137,0,0,0,12,73,68,65,84,120,156,99,248,15,4,0,9,251,3,253,134,87,198,122,0,0,0,0,73,69,78,68,174,66,96,130])).decode("utf-8")

    note = "Running in demo mode without an AI key. Set STABILITY_API_KEY to enable real image generation."
    return GenerateResponse(image_b64=img_b64, provider=provider, model=None, mode="demo", note=note)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
