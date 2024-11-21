# EIS_CNNWithSVM

Actually,in this project i have try many method to solve the problem.
At first,I think  this kinda of classification problem easy to solve for CNN or some other DL method.But Unfortunately when i meet the poor scala of data,i know the reason why previous research just tyr to using ML.
Whatever,i think i should give it a shit and maybe it works?

So in this project,i design a model using CNN with SVM(in my opinion CNN will be a great way to solve the pic problem related, and in others previous research they find SVM is a good choice).i first try extract the featrues from the smoothed curve like peak,length,valley and curvature and typeshit.Then put them all in 1D-CNN,after thousands of(1 hour maybe) parameter and structure fine-tuning,the Accuracy is about 42per.

Then i think that maybe a waste of time by just this poor 502 pieces of data,then i try ResNet but it even worst or i make some mistakes using ResNet.
I also try to using the vae structure in cnn, same, it also not very useful.the accuarcy also about 40 even less.

After this try,i think maybe add the original pic will change(i don't think using just pic is a good idea,even i cannot distinguish just by my eyes),I add the 2D into the CNN-model and cat the previous feature in to linear layer.But it still not have a satisfactory preformance.So I ADD SVM to replace the prediction sectionof the CNN model.

It work, not very well.I am sure this kinda structure does have full potential after tuning and training by great scala of data.But it exhausted all my patience.

SHIT!
