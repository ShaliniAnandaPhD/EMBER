import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_alert_email(wildfire_probability, optimal_resources):
    sender_email = "your_email@example.com"
    receiver_email = "recipient@example.com"
    password = "your_email_password"
    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Wildfire Alert"
    
    body = "Wildfire Probability: {}%\nOptimal Resources: {}".format(wildfire_probability, optimal_resources)
    message.attach(MIMEText(body, "plain"))
    
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender_email, password)
    text = message.as_string()
    server.send_message(message)
    server.quit()

# Main function to send alerts
def main():
    # Load the wildfire probability
    wildfire_probability = np.load("data/wildfire_probability.npy")
    
    # Load the optimal resources
    optimal_resources = np.load("data/optimal_resources.npy")
    
    # Send the alert email
    send_alert_email(wildfire_probability, optimal_resources)
    
    print("Alert email sent successfully.")

if __name__ == "__main__":
    main()
