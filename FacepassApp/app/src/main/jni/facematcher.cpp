#include "facematcher.h"
#include <jni.h>

extern "C" JNIEXPORT jfloat JNICALL
Java_com_artefact_facepass_app_FaceMatcher_compareFaces(JNIEnv *env, jobject instance,
                                                        jstring imagePath1_, jstring imagePath2_) {
    const char *imagePath1 = env->GetStringUTFChars(imagePath1_, 0);
    const char *imagePath2 = env->GetStringUTFChars(imagePath2_, 0);

    env->ReleaseStringUTFChars(imagePath1_, imagePath1);
    env->ReleaseStringUTFChars(imagePath2_, imagePath2);
    return 5.0;
}