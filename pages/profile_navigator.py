import pandas as pd
import streamlit as st
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc, service_pb2, resources_pb2
from clarifai_grpc.grpc.api.status import status_code_pb2
from clarifai.auth.helper import ClarifaiAuthHelper
from clarifai.client import create_stub
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai.urls.helper import ClarifaiUrlHelper
from google.protobuf import json_format
import base64
import numpy as np
import matplotlib.pyplot as plt
import io
from google.protobuf.json_format import MessageToDict
import requests
import cv2
from PIL import Image
from io import BytesIO

auth = ClarifaiAuthHelper.from_streamlit(st)
stub = create_stub(auth)
userDataObject = auth.get_user_app_id_proto()
userDataClarifaiMain= resources_pb2.UserAppIDSet(user_id='clarifai', app_id='main')

# Create a dropdown with options from 'Person-00' to 'Person-14'
person_choices = [f"Person-{str(i).zfill(2)}" for i in range(15)]
selected_person = st.selectbox("Select a person:", person_choices)

if __name__ == "__main__":
    ClarifaiStreamlitCSS.header_css()
    st.title("Clarifai Profile Viewer")
    