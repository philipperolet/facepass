#include <jni.h>

extern "C" JNIEXPORT jfloat JNICALL
Java_com_artefact_facepass_app_FaceMatcher_compareFaces(JNIEnv *env, jobject instance,
                                                        jstring imagePath1_, jstring imagePath2_);