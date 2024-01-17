MYPATH = "C:/Users/Benja/Desktop/array2binaural-1/my_data.csv"
myData <- read.csv(MYPATH)


#%
myData$Subject <- as.factor(myData$Subject)
myData$Stimulus <- as.factor(myData$Stimulus)
myData$Reverb <- as.factor(myData$Reverb)
myData$Rendering <- as.factor(myData$Rendering)
myData$Rotation <- as.factor(myData$Rotation)


aovlist = aov(Spatial ~ Rendering * Reverb * Rotation * Stimulus+ Error((Subject)/(Rendering* Reverb * Rotation*Stimulus)), myData)
summary(aovlist)

aovlist = aov(Timbral ~ Rendering * Reverb  * Rotation * Stimulus+ Error((Subject)/( Rendering* Reverb * Rotation*Stimulus)), myData)
summary(aovlist)


aovlist = manova(cbind(Spatial, Timbral) ~ Rendering * Reverb * Rotation * Stimulus+ Error((Subject)/(Rendering* Reverb * Rotation*Stimulus)), myData)
summary(aovlist, 'Wilks')

