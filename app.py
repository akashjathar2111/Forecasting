import pandas as pd
import streamlit as st
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.title('Indian Import and Export')
choice = ['None','Import','Export']

z = st.sidebar.selectbox('Select Import or Export',choice)

if z == 'None':
    pass
else:
    df1 = pd.read_csv(str(z)+str('_monthly11_15.csv'))
    df2 = pd.read_csv(str(z)+str('_monthly16_21.csv'))
    df = df1.append(df2)
    data = df
    del df
    data['Month']= pd.to_datetime(data['Month'])

    m = data['Commodity'].unique()

    commodity = st.sidebar.selectbox('Select Commodity',m)

    x = ['None','Value(INR)','Quantity']
    feature = st.selectbox('Select Feature',x)
    if feature == "None":
        pass

    elif feature == 'Value(INR)':


        #Total Export value of commodity per year
        value_sum = data[data['Commodity']==commodity].groupby('Month')['value(INR)'].sum()
        value_sum = pd.DataFrame(value_sum)

        #Arima Model
        model = auto_arima(value_sum, start_p=1, start_q=1,
                    max_p=7, max_q=7,
                    m=12,  ######################################           
                    seasonal=True,   
                    start_P=0, 
                    D=None, 
                    trace=True,
                    error_action='ignore',  
                    suppress_warnings=True, 
                    stepwise=True)
        model1_aic = model.aic() 

        #Holt Winter's Model
        ### Holts winter exponential smoothing with additive seasonality and additive trend
        hwe_model_add_add = ExponentialSmoothing(value_sum,seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
        model2_aic = hwe_model_add_add.aic


        ### Holts winter exponential smoothing with multiplicative seasonality and additive trend
        hwe_model_mul_add = ExponentialSmoothing(value_sum,seasonal="mul",trend="add",seasonal_periods=12).fit() 
        model3_aic = hwe_model_mul_add.aic


        st.dataframe({'Model':['ARIMA','Holt Winter with additive seasonality','Holt Winter with multiplicative seasonality'],'AIC':[model1_aic,model2_aic,model3_aic]})
        x = [model1_aic,model2_aic,model3_aic]
        best_model = pd.Series(x).min()

        n = st.sidebar.slider('Forecasted',min_value=1,max_value=50)
        if best_model == model1_aic:
            order = model.order
            seasonal = model.seasonal_order

            X = value_sum.values
            X = X.astype('float32')

            Arima = ARIMA(X,order=order,seasonal_order=seasonal)
            model_fit = Arima.fit()
            forecast = model_fit.forecast(steps = n)
            st.header(f'Value(INR) ARIMA with order={order} and seasnal_order = {seasonal}')
         
            st.line_chart(forecast) 



        elif best_model == model2_aic:
            forecast = hwe_model_add_add.forecast(n)
            st.header('Value(INR) Holts winter exponential smoothing with multiplicative seasonality and additive trend')
            st.line_chart(forecast) 


        elif best_model == model3_aic:
            forecast = hwe_model_mul_add.forecast(n)
            st.header('Value(INR) Holts winter exponential smoothing with multiplicative seasonality and additive trend')
            st.line_chart(forecast)  





    elif feature == 'Quantity':

        #Total Export value of commodity per year
        Qty = data[data['Commodity']==commodity].groupby('Month')['Qty'].sum() 
        unit = data[data['Commodity']==commodity]['Unit'].unique()

        #Arima Model
        model = auto_arima(Qty, start_p=1, start_q=1,
                    max_p=7, max_q=7,
                    m=12,  ######################################           
                    seasonal=True,   
                    start_P=0, 
                    D=None, 
                    trace=True,
                    error_action='ignore',  
                    suppress_warnings=True, 
                    stepwise=True)
        model1_aic = model.aic()  

        #Holt Winter's Model
        ### Holts winter exponential smoothing with additive seasonality and additive trend
        hwe_model_add_add = ExponentialSmoothing(Qty,seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
        model2_aic = hwe_model_add_add.aic

        ### Holts winter exponential smoothing with multiplicative seasonality and additive trend
        hwe_model_mul_add = ExponentialSmoothing(Qty,seasonal="mul",trend="add",seasonal_periods=12).fit() 
        model3_aic = hwe_model_mul_add.aic

        st.dataframe({'Model':['ARIMA','Holt Winter with additive seasonality','Holt Winter with multiplicative seasonality'],'AIC':[model1_aic,model2_aic,model3_aic]})
        n = st.sidebar.slider('Forecasted',min_value=1,max_value=50)
        best_model = pd.Series([model1_aic,model2_aic,model3_aic]).min()
        if best_model == model1_aic:
            order = model.order
            seasonal = model.seasonal_order

            X = Qty.values
            X = X.astype('float32')

            Arima = ARIMA(X,order=order,seasonal_order=seasonal)
            model_fit = Arima.fit()
            forecast = model_fit.forecast(steps = n)
            st.header(f'Quantity (in{unit})')
            
            st.line_chart(forecast)



        elif best_model == model2_aic:
            forecast = hwe_model_add_add.forecast(n)
            st.header(f'Quantity (in{unit})')
            st.line_chart(forecast)


        elif best_model == model3_aic:
            forecast = hwe_model_mul_add.forecast(n) 
            st.header(f'Quantity (in{unit})')
            st.line_chart(forecast) 







    def country(Commodity):
    # for how many country India export this commodity per year
        country_count = data[data['Commodity']==Commodity].groupby('Month')['Country'].count() 

        st.header('Country Count')
        st.line_chart(country_count)
    country(commodity)
