# Text Classification Project (Negative Speech Recognition)

## Overview
Dalam project ini saya melatih model machine learning Logitic Regression (from scratch) untuk memprediksi pakah suatu kalimat(teks) termasuk negative speech atau tidak. Model yang dilatih menghasilkan akurasi sebesar 77%.

## Requirements
Untuk dapat menjalankan program dalam project ini dengan lancar, dibutuhkan setidaknya beberapa library berikut:
- numpy: `1.24.2`
- pandas: `1.5.3`
- regex: `2022.3.15`
- nltk: `3.7`
- tqdm: `4.64.1`
- sklearn: `0.0.post1`
- Sastrawi: `1.0.1`
- flask: `1.1.2`
- flask-restx: `1.0.6`

Untuk menginstallnya sekaligus, saya sudah menyediakan file requirements.txt pada repository ini. Anda dapat menginstallnya dengan menjalankan perintah `pip install -r requirements.txt`.

## Dataset
Pada project ini saya menggunakan data teks berupa post/status media sosial yang dikumpulkan dari berbagai source (twitter, kaskus, dll). Terima kasih untuk IndoNLP dan akun GitHub ahmadizzan telah membagikan data ini. (Data dapat diakses [di sini](https://github.com/ahmadizzan/netifier))

## API Project
Pada project ini saya mmembuat sebuah ML API dalam Flask yang sudah terintegrasi dengan swagger. Anda dapat mencobanya dengan menjalankan perintah `python app.py` pada direktori api_project.
