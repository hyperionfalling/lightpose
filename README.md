# lightpose
参加比赛代码，轻量级姿态识别模型
环境：mini-batch size 16 GPU 1660Ti
学习率 5.4 * 10 ^(-4),逐步衰减
最好效果
	      Head	  Shoulder	Elbow	  Wrist	  Hip	    Knee	  Ankle	  Mean
  56000	77.55	  91.51	    70.14	  69.06	  80.77	  69.82	  62.40	  75.39
模型结构
Rethinking on Multi-Stage Networks for Human Pose Estimation
采用Ghost net作为卷积层，引入轻量级残差注意力层
原模型（1-stage）大小为290M，采用以上改进后模型大小为90M。
