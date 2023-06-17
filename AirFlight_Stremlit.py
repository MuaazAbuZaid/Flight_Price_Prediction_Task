import streamlit as st
import pandas as pd
import joblib


Inputs = joblib.load("Inputs.pkl")
Model = joblib.load("XGB_Model")

def prediction(Airline, Source, Destination, Month, Day_Name, Total_Stops, Duration):
    test_df = pd.DataFrame(columns=Inputs)
    test_df.at[0,"Airline"] = Airline
    test_df.at[0,"Source"] = Source
    test_df.at[0,"Destination"] = Destination
    test_df.at[0,"Month"] = Month
    test_df.at[0,"Day_Name"] = Day_Name
    test_df.at[0,"Total_Stops"] = Total_Stops
    test_df.at[0,"Duration"] = Duration
    st.dataframe(test_df)
    result = Model.predict(test_df)[0]
    return result


    
def main():
    st.title("Airflight Price Prediction")
    Airline = st.selectbox("Airline" , ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet',
       'Multiple carriers', 'GoAir', 'Vistara', 'Air Asia',
       'Vistara Premium economy', 'Jet Airways Business',
       'Multiple carriers Premium economy', 'Trujet'])
    Source = st.selectbox("Source" , ['Banglore', 'Kolkata', 'Delhi', 'Chennai', 'Mumbai'])
    Destination = st.selectbox("Destination" , ['New Delhi', 'Banglore', 'Cochin', 'Kolkata', 'Delhi', 'Hyderabad'])
    Month = st.selectbox("Month" ,['March', 'May', 'June', 'April'] )
    Day_Name = st.selectbox("Day_Name" , ['Sunday', 'Wednesday', 'Friday', 'Monday', 'Tuesday', 'Saturday','Thursday'])
    Duration = st.slider("Duration(mins)" , min_value=30, max_value=5000, value=0, step=1)
    Total_Stops = st.slider("Total_Stops" , min_value=0, max_value=6, value=0, step=1)
    
    if st.button("predict"):
        result = prediction(Airline, Source, Destination, Month, Day_Name, Total_Stops, Duration)
        st.text(f"Ticket Price is about {result}")
        
if __name__ == '__main__':
    main()    
 
