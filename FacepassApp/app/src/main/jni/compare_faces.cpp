// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*
    This is an example illustrating the use of the deep learning tools from the dlib C++
    Library.  In it, we will show how to do face recognition.  This example uses the
    pretrained dlib_face_recognition_resnet_model_v1 model which is freely available from
    the dlib web site.  This model has a 99.38% accuracy on the standard LFW face
    recognition benchmark, which is comparable to other state-of-the-art methods for face
    recognition as of February 2017. 
    
    In this example, we will use dlib to do face clustering.  Included in the examples
    folder is an image, bald_guys.jpg, which contains a bunch of photos of action movie
    stars Vin Diesel, The Rock, Jason Statham, and Bruce Willis.   We will use dlib to
    automatically find their faces in the image and then to automatically determine how
    many people there are (4 in this case) as well as which faces belong to each person.
    
    Finally, this example uses a network with the loss_metric loss.  Therefore, if you want
    to learn how to train your own models, or to get a general introduction to this loss
    layer, you should read the dnn_metric_learning_ex.cpp and
    dnn_metric_learning_on_images_ex.cpp examples.
*/

#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <chrono>
//#include "compare_faces.h2"

using namespace dlib;
using namespace std;

// ----------------------------------------------------------------------------------------

// The next bit of code defines a ResNet network.  It's basically copied
// and pasted from the dnn_imagenet_ex.cpp example, except we replaced the loss
// layer with loss_metric and made the network somewhat smaller.  Go read the introductory
// dlib DNN examples to learn what all this stuff means.
//
// Also, the dnn_metric_learning_on_images_ex.cpp example shows how to train this network.
// The dlib_face_recognition_resnet_model_v1 model used by this example was trained using
// essentially the code shown in dnn_metric_learning_on_images_ex.cpp except the
// mini-batches were made larger (35x15 instead of 5x5), the iterations without progress
// was set to 10000, and the training dataset consisted of about 3 million images instead of
// 55.  Also, the input layer was locked to images of size 150.
template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;


// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
);

// ----------------------------------------------------------------------------------------

void compare_faces(matrix<rgb_pixel> img1, matrix<rgb_pixel> img2) {

  auto start = std::chrono::system_clock::now();
  
  // The first thing we are going to do is load all our models.  First, since we need to
  // find faces in the image we will need a face detector:
  frontal_face_detector detector = get_frontal_face_detector();
  // We will also use a face landmarking model to align faces to a standard pose:  (see face_landmark_detection_ex.cpp for an introduction)
  shape_predictor sp;
  deserialize("shape_predictor_5_face_landmarks.dat") >> sp;
  // And finally we load the DNN responsible for face recognition.
  anet_type net;
  deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
  
  // Run the face detector on the image of our action heroes, and for each face extract a
  // copy that has been normalized to 150x150 pixels in size and appropriately rotated
  // and centered.
  auto load_time = std::chrono::system_clock::now();
  std::vector<matrix<rgb_pixel>> faces1;
  for (auto face : detector(img1))
    {
      auto shape = sp(img1, face);
      matrix<rgb_pixel> face_chip;
      extract_image_chip(img1, get_face_chip_details(shape,150,0.25), face_chip);
      faces1.push_back(move(face_chip));
    }
  auto detection_time = std::chrono::system_clock::now();
  std::vector<matrix<float,0,1>> face_descriptors1 = net(faces1);
  auto embedding_time = std::chrono::system_clock::now();
  //  matrix<float,0,1> fd1 = mean(mat(net(jitter_image(faces1[0]))));
  auto embedding_jitter = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed1 = load_time - start;
  std::chrono::duration<double> elapsed2 = detection_time - load_time;
  std::chrono::duration<double> elapsed3 = embedding_time - detection_time;
  std::chrono::duration<double> elapsed4 = embedding_jitter - embedding_time;

  cout << elapsed1.count() << endl;
  cout << elapsed2.count() << endl;
  cout << elapsed3.count() << endl;
  cout << elapsed4.count() << endl;

  std::vector<matrix<rgb_pixel>> faces2;
  for (auto face : detector(img2))
    {
      auto shape = sp(img2, face);
      matrix<rgb_pixel> face_chip;
      extract_image_chip(img2, get_face_chip_details(shape,150,0.25), face_chip);
      faces2.push_back(move(face_chip));
    }
  if (faces1.size() == 0 || faces2.size() == 0)
    {
        cout << "No faces found in image!" << endl;
        return;
    }

    // This call asks the DNN to convert each face image in faces into a 128D vector.
    // In this 128D vector space, images from the same person will be close to each other
    // but vectors from different people will be far apart.  So we can use these vectors to
    // identify if a pair of images are from the same person or from different people.  

  std::vector<matrix<float,0,1>> face_descriptors2 = net(faces2);
  matrix<float,0,1> fd1 = face_descriptors1[0];
  matrix<float,0,1> fd2 = mean(mat(net(jitter_image(faces2[0]))));
  float face_distance = length(fd1 - fd2);
  cout << "Face distance" << face_distance << endl;
  if (face_distance < 0.6) {
    cout << "SAME FACES!" << endl;
  }
  else {
    cout << "DIFFERENT FACES!" << endl;
  }
}

int main(int argc, char** argv) try
{
    if (argc != 3)
    {
        cout << "Run this example by invoking it like this: " << endl;
        cout << "   ./compare_faces f1.jpg f2.jpg" << endl;
        cout << endl;
        cout << "You will also need to get the face landmarking model file as well as " << endl;
        cout << "the face recognition model file.  Download and then decompress these files from: " << endl;
        cout << "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2" << endl;
        cout << "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2" << endl;
        cout << endl;
        return 1;
    }
    matrix<rgb_pixel> img1;
    load_image(img1, argv[1]);
    matrix<rgb_pixel> img2;
    load_image(img2, argv[2]);

    // Display the raw image on the screen
    image_window win1(img1); 
    image_window win2(img2);


    compare_faces(img1, img2);
    
    cout << "hit enter to terminate" << endl;
    cin.get();
}
catch (std::exception& e)
{
    cout << e.what() << endl;
}

// ----------------------------------------------------------------------------------------

std::vector<matrix<rgb_pixel>> jitter_image(
    const matrix<rgb_pixel>& img
)
{
    // All this function does is make 100 copies of img, all slightly jittered by being
    // zoomed, rotated, and translated a little bit differently. They are also randomly
    // mirrored left to right.
    thread_local dlib::rand rnd;

    std::vector<matrix<rgb_pixel>> crops; 
    for (int i = 0; i < 100; ++i)
        crops.push_back(jitter_image(img,rnd));

    return crops;
}
