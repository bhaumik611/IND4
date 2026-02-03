# 🏠 House Price Prediction Web Application

A full-stack web application that predicts house prices based on property features such as city, BHK, area, construction status, and resale information. The backend uses a machine learning model built with Scikit-learn, while the frontend provides a user-friendly interface for predictions.

---

## 📌 Project Overview

This project aims to estimate house prices (in Lakhs) using historical housing data and machine learning techniques. The application consists of:

- **Frontend**: Collects user input and displays predicted prices
- **Backend (Flask API)**: Handles requests, processes data, and returns predictions
- **Machine Learning Model**: Ridge Regression with feature scaling

---

## ✨ Features

- Predict house prices in real time
- City-based price differentiation using **City Tier classification**
- Machine Learning model with feature scaling
- REST API for predictions
- Health check and test endpoints
- CORS enabled for frontend-backend communication

---

## 🧠 Machine Learning Details

- **Algorithm**: Ridge Regression
- **Preprocessing**:
  - Log transformation of target prices
  - Feature scaling using `StandardScaler`
- **Target Variable**: `TARGET(PRICE_IN_LACS)`
- **Model Persistence**: Saved using `joblib`

### 🔢 Input Features

| Feature | Description |
|------|-----------|
| UNDER_CONSTRUCTION | 1 if under construction, else 0 |
| RERA | 1 if RERA approved, else 0 |
| BHK_NO. | Number of bedrooms |
| SQUARE_FT | Area in square feet |
| READY_TO_MOVE | 1 if ready to move |
| RESALE | 1 if resale property |
| City_Tier | Tier 1 (Metro) or Tier 2 (Non-metro) |

### 🏙️ City Tier Logic
- **Tier 1 (Metro cities)**: Mumbai, Delhi, Bengaluru, Chennai, Hyderabad, Pune, Kolkata, Ahmedabad
- **Tier 2**: All other cities

---

## 🛠️ Tech Stack

### Frontend
- HTML
- CSS
- JavaScript

### Backend
- Python
- Flask
- Flask-CORS

### Machine Learning
- Pandas
- NumPy
- Scikit-learn
- Joblib

---
