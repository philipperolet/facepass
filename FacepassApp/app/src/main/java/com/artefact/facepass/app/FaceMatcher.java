package com.artefact.facepass.app;

public class FaceMatcher {

    /*static {
        System.loadLibrary("face-embed");
    }*/

    public native float compareFaces(String imagePath1, String imagePath2);

}
