import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
file = './model/sample_writer.csv'

#AVE= [[] for i in range(40)]
#Frames= [[] for i in range(40)]

AVE= [[] for i in range(1)]
Frames= [[] for i in range(1)]
#print(AVE)

for m in range(1):
  #file = './model/Origin_200_goals_Singlebranch/8threds/'+str(m)+'sample_writer.csv'
  file = './model/Transformer/'+str(m)+'samplewriter.csv'
  file = 'model/Transformer_word2vec/20scene/samplewriter.csv'
#  file = './model/Transformer/test/'+str(m)+'samplewriter.csv'
  #file = './model/Origin_100_goals_Singlebranch/10_Threds/'+str(m)+'sample_writer.csv'
  #file = './model/Target_Driven_Original_100_goals/'+str(m)+'sample_writer.csv'
  #file = './model/SMT_100_target_only_target/'+str(m)+'sample_writer.csv'
  #file = './model/test/'+str(m)+'sample_writer.csv'
  #file = './model/SMT_200_goals_Concat/v1/'+str(m)+'sample_writer.csv'
  #file = './model/SMT_200_goals_Sum/'+str(m)+'sample_writer.csv'
  #file = './model/SMT_200_target/'+str(m)+'sample_writer.csv'
  #file = './model/Target_Driven_Original_200_target/'+str(m)+'sample_writer.csv'
  #file = './model/Original_200_target/'+str(m)+'sample_writer.csv'
  #file = './model/SMT_100_goals/TEST2/'+str(m)+'sample_writer.csv'
  #file = './model/SMT_100_goals/'+str(m)+'sample_writer.csv'
  f = open(file,'r')
  rows = csv.reader(f)
  j = 1
  Sum = 0
  #Ave = list()
  #Frames = list()
  count = 0
  for row in rows:
    if len(row)== 1:
      pass
    elif int(row[1]) >20000000:
      pass
    else:
      if int(row[1]) < j * 200:
        Jud = True
      else:
        Jud = False
      if Jud:
        Sum += int(row[2])
        count += 1
      else:
        JUD = True
        while JUD:
          if int(row[1]) < j *200:
            JUD = False
          else:
            if count == 0:
              AVE[m].append(np.nan)
            else:
              AVE[m].append(Sum/count)
              Sum = 0
              count = 0
            Frames[m].append(j* 0.5)
            j += 1
        Sum += int(row[2])
        count += 1
        #if int(row[1]) > 10000000:
        #    break
  f.close()
import numpy as np
print(AVE)
min_d = 100000
for i in range(len(AVE)):
  print(len(AVE[i]))
  if len(AVE[i]) < min_d:
    min_d = len(AVE[i])
for i in range(len(AVE)):
  if len(AVE[i]) > min_d:
    n = len(AVE[i])-min_d
    del AVE[i][-n:]
for i in range(len(Frames)):
  if len(Frames[i]) > min_d:
    n = len(Frames[i])-min_d
    del Frames[i][-n:]

for i in range(len(AVE)):
  print(len(AVE[i]))
AVE = np.array(AVE)
#print(AVE)
Ave = np.nanmean(AVE, axis=0)
#print(Ave)
Frames = np.mean(Frames, axis=0)

fig = plt.figure()
plt.title("learning history")
#plt.xlabel("training frames")
plt.xlabel("training steps")
plt.ylabel("Average length")
plt.grid()
y = Ave
x = Frames
plt.plot(x,y, color="m")
plt.yscale('log')
#fig.savefig("./data/img.png")
#_,y_max = plt.ylim()
#plt.ylim([100,y_max])
#plt.ylim([100,5000])
#plt.ylim([100,1000])
#plt.ticklable_format(style='sci',axis='x',scilimits=(0,0))
#plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda d,pos: int(d/1000000)))
plt.xlabel('Frames (val in millions)')
plt.show()
plt.close()
