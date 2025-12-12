# create venv
``python -m venv venv``

# run API server and the streamlit app
```
source venv/bin/activate
uvicorn main:app --reload
streamlit run streamlit_app.py
```


# To clean the user table and media folder
```
sudo -u postgres psql -d face_db -c "TRUNCATE TABLE users RESTART IDENTITY CASCADE;"
rm -f ~/apps/face_api/media/users/*.jpg
```
