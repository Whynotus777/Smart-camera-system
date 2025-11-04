import smtplib

def send_email_alert(subject, message):
    sender = "your@email.com"
    receiver = "manager@store.com"
    msg = f"Subject: {subject}\n\n{message}"
    with smtplib.SMTP("smtp.example.com", 587) as server:
        server.starttls()
        server.login(sender, "password")
        server.sendmail(sender, receiver, msg)
