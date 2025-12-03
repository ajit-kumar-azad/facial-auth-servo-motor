import os
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy.orm import Session
import numpy as np
import face_recognition
import io
from PIL import Image

from database import SessionLocal
from models import User

MEDIA_ROOT = os.path.abspath("media")
os.makedirs(os.path.join(MEDIA_ROOT, "users"), exist_ok=True)

app = FastAPI(title="Face Recognition API", version="1.1.0")

# Serve /media/*
app.mount("/media", StaticFiles(directory=MEDIA_ROOT), name="media")


# Dependency for DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Pydantic schemas
class UserOut(BaseModel):
    id: int
    name: str
    photo_url: Optional[str] = None   # <--- NEW

    class Config:
        orm_mode = True

class UserWithEmbedding(UserOut):
    embedding: List[float]


class IdentifyResult(BaseModel):
    name: str
    user_id: int | None = None
    distance: float | None = None

def photo_url_from_path(request: Request, photo_path: str | None) -> str | None:
    if not photo_path:
        return None
    # photo_path is like "users/123.jpg"
    return str(request.base_url)[:-1] + "/media/" + photo_path.replace("\\", "/")

def save_user_photo(user_id: int, file_bytes: bytes) -> str:
    # Always save as JPEG for consistency
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    rel_path = f"users/{user_id}.jpg"
    abs_path = os.path.join(MEDIA_ROOT, rel_path)
    img.save(abs_path, format="JPEG", quality=90)
    return rel_path

def get_face_embedding(file_bytes: bytes) -> np.ndarray:
    image = face_recognition.load_image_file(io.BytesIO(file_bytes))
    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        raise ValueError("No face detected in the image")
    # You could enforce 1 face only if you want:
    # if len(encodings) > 1: raise ValueError("Multiple faces detected")
    return encodings[0]


@app.post("/users", response_model=UserOut, summary="Create a user with face embedding")
async def create_user(
    request: Request,
    name: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Image must be JPEG or PNG")

    file_bytes = await image.read()

    try:
        embedding = get_face_embedding(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 1) Create user to get ID
    user = User(name=name, embedding=embedding.tolist())
    db.add(user)
    db.commit()
    db.refresh(user)

    # 2) Save photo to disk
    rel_path = save_user_photo(user.id, file_bytes)

    # 3) Update row with photo_path
    user.photo_path = rel_path
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserOut(
        id=user.id,
        name=user.name,
        photo_url=photo_url_from_path(request, user.photo_path)
    )


FACE_MATCH_THRESHOLD = 0.6  # tune as needed

@app.post("/identify", response_model=IdentifyResult, summary="Identify face from database")
async def identify_user(
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Image must be JPEG or PNG")

    file_bytes = await image.read()

    # Get embedding for input image
    try:
        query_embedding = get_face_embedding(file_bytes)
    except ValueError as e:
        # No face found => explicit response
        raise HTTPException(status_code=400, detail=str(e))

    # Fetch all users (for large DB, you'd want vector index / pgvector)
    users: List[User] = db.query(User).all()

    if not users:
        return IdentifyResult(name="UNKNOWN")

    best_user = None
    best_distance = None

    for user in users:
        embedding = np.array(user.embedding, dtype=np.float32)
        dist = np.linalg.norm(embedding - query_embedding)

        if best_distance is None or dist < best_distance:
            best_distance = dist
            best_user = user

    if best_distance is not None and best_distance <= FACE_MATCH_THRESHOLD:
        return IdentifyResult(
            name=best_user.name,
            user_id=best_user.id,
            distance=float(best_distance),
        )

    return IdentifyResult(name="UNKNOWN")


@app.delete("/users/{user_id}", summary="Delete user by ID")
def delete_user(
    user_id: int,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()
    return {"detail": "User deleted successfully"}


@app.get("/users", response_model=List[UserOut], summary="List users")
def list_users(
    request: Request,
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    users = (
        db.query(User)
        .order_by(User.id)
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [
        UserOut(
            id=u.id,
            name=u.name,
            photo_url=photo_url_from_path(request, u.photo_path)
        )
        for u in users
    ]

@app.get("/users_full", response_model=List[UserWithEmbedding], summary="List users with embeddings")
def list_users_full(request: Request, db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.id).all()
    out = []
    for u in users:
        out.append(UserWithEmbedding(
            id=u.id,
            name=u.name,
            embedding=u.embedding,
            photo_url=photo_url_from_path(request, u.photo_path)
        ))
    return out