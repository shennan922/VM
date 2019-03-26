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
rec = ['keyong.yu@accenture.com','nan.a.shen@accenture.com','miao.d.li@accenture.com']
subject = 'Accuracy report'

logpath = '../train_result.txt'
tf.app.flags.DEFINE_string(
    'path', '', 'result directory ')
tf.app.flags.DEFINE_string(
    'create_pb_step', '', 'create_pb_step ')
tf.app.flags.DEFINE_string(
    'data_dir', '', 'images directory ')
tf.app.flags.DEFINE_string(
    'results_folder', '', 'result directory ')
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
create_pb_step = FLAGS.create_pb_step
data_dir = FLAGS.data_dir
results_folder = FLAGS.results_folder
details = 'Step:' + create_pb_step + '\n' + 'Data_dir:' + data_dir + '\n' +'Results_dir:' + results_folder

total = details + '\n' + '\n' + log + '\n' + '\n' + data

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
