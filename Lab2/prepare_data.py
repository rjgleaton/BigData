import pandas as pd
# compound,sg,Eg (eV),m*b (vb-cb),Nb (vb-cb),kL,"mob (h,e)","beta (p,n)",natoms,density,volume,bulkmod,"m*DOS(vb,cb)",avgcn,gamma

with open("thermal2700.csv") as ofile:
	lines=ofile.readlines()

	records=[]
	for x in lines[1:]:
		items=x.split(",")
		print(items)
		
		break

exit()


df=pd.read_csv("thermal2700.csv")
print(df.head())

df1=df['m*b (vb-cb)']
vb=[]
cb=[]
for x in df1:
	print(x.split(","))
	vb.append(float(x.split(",")[0]))	
	cb.append(float(x.split(",")[1]))
print(cb)
# exit()

df1=df['Nb (vb-cb)']
Nvb=[]
Ncb=[]
for x in df1:
	Nvb.append(x.split(",")[0])	
	Ncb.append(x.split(",")[1])
print(Ncb)

df1=df['mob (h,e)']
h=[]
e=[]
for x in df1:
	h.append(x.split(",")[0])	
	e.append(x.split(",")[1])
print(h)

df1=df['beta (p,n)']
p=[]
n=[]
for x in df1:
	p.append(x.split(",")[0])	
	n.append(x.split(",")[1])
print(p)

print(len(vb),df.shape)
df2=pd.concat([df,pd.DataFrame({'vb':vb,'cb':cb})])
print(df2.head())
# print(vb)
# print(df2['cb'])
# for x in df2['cb']:
# 	print(x)