source venv/bin/activate
uvicorn main:app --reload
streamlit run streamlit_app.py


# To cleand the user table and media folder
sudo -u postgres psql -d face_db -c "TRUNCATE TABLE users RESTART IDENTITY CASCADE;"
rm -f ~/apps/face_api/media/users/*.jpg