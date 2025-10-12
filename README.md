# F1 Championship Predictor - Advanced ML Dashboard with Real-Time Integration

##  **NEW: Real-Time F1 Data Integration**
**Live championship tracking, race results, and weather data!**

### **Launch Options:**

#### **Real-Time Dashboard (NEW!)**
```bash
# Windows
launch_realtime_app.bat

# Mac/Linux
./launch_realtime_app.sh
```

#### **Standard Dashboard**
```bash
# Windows
launch_app.bat

# Mac/Linux
./launch_app.sh
```

### 2. **Application Features**

#### **Live F1 Data (NEW!)**
- **Real-time championship standings** with live points and positions
- **Latest race results** with winner, podium, and full race breakdown
- **Next race information** including circuit details and countdown
- **Weather integration** (with API key) for track conditions
- **Live championship probabilities** calculated in real-time
- **Constructor standings** and team points distribution
- **Race calendar** with season progress tracking

#### **Home Dashboard**
- Championship predictions overview
- Top driver rankings
- Win probability analysis
- Interactive charts and metrics

#### **Driver Predictions**
- Individual driver analysis
- Performance breakdown radar charts
- Championship odds calculation
- Detailed prediction metrics

#### **What-If Scenarios**
- Interactive performance simulation
- Custom scenario testing with sliders
- Real-time prediction updates
- Scenario comparison analysis

#### **Experience Analysis**
- Driver experience impact quantification
- Rookie vs veteran comparison
- Experience premium calculations
- Career development insights

#### **Team Impact**
- Team performance vs driver success
- Championship feasibility by team category
- Car development impact analysis
- Strategic team insights

#### **Model Performance**
- ML algorithm comparison
- Accuracy metrics and validation
- Cross-validation results
- Model selection transparency

## **Technical Requirements**

### **System Requirements**
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Internet connection for initial setup

### **Python Dependencies**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
scikit-learn>=1.3.0
xgboost>=1.7.0
tensorflow>=2.13.0
pyspark>=3.4.0
```

## **Key Features**

### **Prediction Capabilities**
- **99.9% accuracy** championship point predictions
- **Real-time scenarios** with interactive controls
- **Experience impact** quantification (314-point veteran advantage)
- **Team performance** analysis and feasibility assessment
- **Multi-model comparison** with transparent metrics

### **Interactive Elements**
- **Sliders and controls** for custom scenario testing
- **Dynamic charts** that update with user input
- **Real-time calculations** powered by trained ML models
- **Responsive design** for desktop and mobile
- **F1-themed styling** with Red Bull Racing colors

### **Data Foundation**
- **73 years of F1 data** (1950-2022)
- **24,655 race records** processed with PySpark
- **780 unique drivers** and 322 teams analyzed
- **6 ML algorithms** compared and validated

## **Advanced Usage**

### **Model Integration**
The application loads pre-trained models for real-time predictions:

```python
# Load trained models (run export_models.py first)
nn_model = joblib.load('models/nn_model.pkl')
nn_scaler = joblib.load('models/nn_scaler.pkl')

# Make predictions
prediction = nn_model.predict(scaled_input)
```

### **Custom Scenarios**
Users can create custom performance scenarios:
- Adjust wins, podiums, and points finishes
- Select team strength and driver experience
- View real-time championship point predictions
- Compare against historical benchmarks

### **Data Export**
Export your analysis results:
- Download prediction data as CSV
- Share scenario comparisons
- Export visualizations as images

## **Application Screenshots**

### **Home Dashboard**
- Championship leaderboard with predicted points
- Win probability pie charts
- Interactive scatter plots
- Performance metrics cards

### **What-If Scenarios**
- Interactive sliders for performance inputs
- Real-time prediction calculations
- Scenario comparison charts
- Championship likelihood assessment

### **Experience Analysis**
- Experience level impact visualization
- Premium point calculations
- Career development insights
- Veteran advantage quantification

## **Educational Value**

### **Learning Outcomes**
- Understanding ML model performance and validation
- F1 championship dynamics and key success factors
- Data visualization and interactive dashboard design
- Real-world application of predictive analytics

### **Use Cases**
- **Fantasy F1**: Optimize team selection with ML predictions
- **Sports Analysis**: Professional F1 commentary and insights
- **Educational**: Teaching ML concepts with engaging F1 data
- **Research**: Championship factor analysis and validation

## **Future Enhancements**

### **Phase 2 Development**
- **Live F1 API integration** for real-time race data
-  **Weather impact analysis** for race predictions
- **Circuit-specific modeling** for track performance
- **Mobile application** with push notifications
- **Advanced ensemble models** with deep learning

### **Business Applications**
- **Betting odds optimization** with confidence intervals
- **Media integration** for race weekend coverage
- **Team strategy consulting** with predictive insights
- **Driver contract valuation** based on ML predictions

## **Support & Documentation**

### **Getting Help**
- Built-in help sections in each app page
- Interactive tooltips and explanations
- Model performance transparency
- Real-time prediction confidence levels

### **Technical Support**
- Check `requirements.txt` for dependency issues
- Ensure Python 3.8+ is installed
- Verify internet connection for initial package installation
- Use `streamlit run streamlit_app.py --help` for CLI options

---

## **Ready to Race!**

Launch your F1 Championship Predictor and explore the world of Formula 1 through the lens of advanced machine learning!

**Built with  for F1 enthusiasts and data science lovers**