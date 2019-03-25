import tensorflow as tf

tf.app.flags.DEFINE_string(
    'num_classes', '', 'num_classes ')
tf.app.flags.DEFINE_string(
    'pipeline_config_path', '', 'pipeline_config directory ')
tf.app.flags.DEFINE_string(
    'pipeline_config_path_new', '', 'pipeline_config directory ')
tf.app.flags.DEFINE_string(
    'scales', '', 'scales value ')
#tf.app.flags.DEFINE_string(
#    'aspect_ratios', '', 'aspect_ratios value ')
tf.app.flags.DEFINE_string(
    'train_path', '', 'train directory ')
tf.app.flags.DEFINE_string(
    'eval_path', '', 'eval directory ')
tf.app.flags.DEFINE_string(
    'label_map_path', '', 'eval directory ')
tf.app.flags.DEFINE_string(
    'input_folder', '', 'input_folder directory ')

FLAGS = tf.app.flags.FLAGS

def initSpace(num):
    space =''
    for i in range(num):
        space+=" "
    return space;
def initScalorratios(p ,scales,type):
    space =''
    for i in scales:
        space = space + initSpace(p)+ type+ " :" + i + "\n"
    return space;

f = open(FLAGS.pipeline_config_path)
f1 = open(FLAGS.pipeline_config_path_new, 'w')
line = f.readline()
#scales = ['1', '1', '0.4982270591262632']
#aspect_ratios = "'1.5364526729519685', '0.2778944985034132', '0.4982270591262632'"
#scales = list(FLAGS.scales)

filename=FLAGS.input_folder + '/scales.txt'
with open(filename) as read_file:
    for line_num,eachline in enumerate(read_file):
        if line_num == 0:
            aspect_ratios = eachline
            aspect_ratios = aspect_ratios.replace("[","").replace(" ","").replace("]","").replace("\n","").split(',')
            #print(aspect_ratios)
        
        #if line_num == 1:
        #    scales = eachline
        #    scales = scales.replace("[","").replace("]","").split(' ')
         #   scales_last = list(filter(None, scales))
         #   print(scales_last)
        
scales = FLAGS.scales
scales = scales.replace("[","").replace("]","").split(',')
scales_last = list(filter(None, scales))
#print(scales_last)

endScales = False
endRatios = False
train_input_reader = False
eval_input_reader = False
while line:
    str = line.strip();
    if not str.startswith('#'):
        if str.startswith('num_classes'):
            p = line.index('num_classes')
            line = initSpace(p) + "num_classes: " + FLAGS.num_classes+"\n"
        if str.startswith('train_input_reader'):
            train_input_reader = True
            eval_input_reader = False
        if str.startswith('eval_input_reader'):
            train_input_reader = False
            eval_input_reader = True
        if str.startswith('input_path'):
            if train_input_reader:
                p = line.index('input_path')
                #print("train")
                line = initSpace(p)+"input_path: "+"\""+FLAGS.train_path+"\"\n"
            if eval_input_reader:
                p = line.index('input_path')
                #print("eval")
                line = initSpace(p) + "input_path: " +"\""+ FLAGS.eval_path+"\"\n"
        if str.startswith('label_map_path'):
            p = line.index('label_map_path')
            line = initSpace(p) + "label_map_path: " +"\""+ FLAGS.label_map_path+"\"\n"
        
        if str.startswith('scales') or str.startswith('min_scale') or str.startswith('max_scale'):
           p=line.find('scales')
           if p==-1:
              p = line.find('min_scale')
           if p==-1:
              p = line.find('max_scale')
#           #print(str)
           if not endScales:
                line =initScalorratios(p,scales_last,'scales')
                endScales = True
           else:
                line=''
        if str.startswith('aspect_ratios'):
           #print(str)
           p = line.index('aspect_ratios')
           if not endRatios:
                line = initScalorratios(p, aspect_ratios, 'aspect_ratios')
                endRatios = True
           else:
                line=''
        
    f1.write(line)
    line = f.readline()

f.close()

