import os
import fnmatch
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import tensorflow as tf

mail_host = "smtp.gmail.com"

user = "IrisLi0602@gmail.com"
passwd = "lnyxlm5865865"

send = 'IrisLi0602@gmail.com'
rec = ['IrisLi0602@gmail.com']
subject = 'Accuracy report'

logpath = '../train_result.txt'
tf.app.flags.DEFINE_string(
    'path', '', 'result directory ')
FLAGS = tf.app.flags.FLAGS

data = ''
log = ''
total = ''
with open(os.path.join(logpath)) as src:
    log = log + ''.join(src.readlines())
    src.close()
    
with open(os.path.join(FLAGS.path)) as src:
    data = data + ''.join(src.readlines())
    src.close()

# print data
total = log + '\n' + '\n' + data

msg = MIMEText(total, 'plain', 'utf-8')
msg['Subject'] = Header(subject, 'utf-8')
msg['From'] = send
msg['To'] = ",".join(rec)
try:
    smtp = smtplib.SMTP()
    smtp.connect(mail_host)
    smtp.starttls()
    smtp.login(user, passwd)
    smtp.sendmail(send, rec, msg.as_string())
    print("success")
    smtp.quit()
except smtplib.SMTPException as e:
    print(e)

#if os.path.exists(logpath) == True:
#    os.remove(logpath)
