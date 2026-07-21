import streamlit as st

st.title("Greetings 👋 ¡Hola! 👋 Bonjour")

st.text("")

st.subheader(
"""
Hi, I'm Sujal Adhikari, a Computer Science and Mathematics student at Caldwell University with a passion for Artificial Intelligence, Machine Learning, and Data Science.

This project represents my journey of applying machine learning to real-world financial problems. Through this stock volatility prediction system, I explored how different approaches — including traditional statistical models, machine learning models, and deep learning architectures — can be used to understand and forecast market behavior.

Building this project helped me strengthen my skills in data preprocessing, feature engineering, model development, evaluation, and deploying machine learning applications. From collecting and transforming financial data to comparing models such as XGBoost, RNN, LSTM, and GARCH, every step provided valuable experience in solving practical data-driven challenges.

Beyond this project, I am also involved in research focused on Federated Learning and privacy-preserving machine learning for network intrusion detection. These experiences have deepened my interest in developing intelligent systems that can solve meaningful problems while addressing real-world constraints.

There have been many challenges throughout this journey — debugging models, understanding complex algorithms, and improving performance — but each challenge has helped me become a better developer and researcher.

This is only the beginning of my journey in AI and Data Science. I am excited to continue learning, building impactful projects, and transforming data into meaningful insights.
"""
)

st.text("")

st.divider()

st.subheader("Thank You 🙏 Gracias! 🙏 Merci")


# Sidebar
st.sidebar.title("Meet the Developer")

st.sidebar.header("Sujal Adhikari")

st.sidebar.write("Computer Science & Mathematics Student")
st.sidebar.write("AI | Machine Learning | Data Science")

st.sidebar.text("")

st.sidebar.write(
"""
Welcome to my project portfolio!

I enjoy building machine learning applications, exploring artificial intelligence research, and developing data-driven solutions to real-world problems.

Currently, my interests include machine learning, deep learning, federated learning, financial modeling, and deploying AI systems.
"""
)

st.sidebar.markdown(
    "[GitHub](https://github.com/suzaladhikari)",
    unsafe_allow_html=True
)

st.sidebar.markdown(
    "[LinkedIn](https://www.linkedin.com/in/sujal-adhikari/)",
    unsafe_allow_html=True
)

st.sidebar.write("sujal.adhikari.ds@gmail.com")