/*
 * Get one gray image using libdc1394 and store it as portable gray map
 *    (pgm). Based on 'samplegrab' from Chris Urmson
 *
 * Written by Damien Douxchamps <ddouxchamps@users.sf.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */

#include <stdio.h>
#include <stdint.h>
#include <dc1394/dc1394.h>
#include <stdlib.h>
#include <inttypes.h>

#ifndef _WIN32
#include <unistd.h>
#endif

uint8_t * rgb;

/*-----------------------------------------------------------------------
 *  Releases the cameras and exits
 *-----------------------------------------------------------------------*/
void cleanup_and_exit(dc1394camera_t *camera)
{
    dc1394_video_set_transmission(camera, DC1394_OFF);
    dc1394_capture_stop(camera);
    dc1394_camera_free(camera);
    free(rgb);
    exit(1);
}

int main(int argc, char *argv[])
{
    FILE* imagefile;
    dc1394camera_t *camera;
    int i;
    dc1394featureset_t features;
    dc1394framerates_t framerates;
    dc1394video_modes_t video_modes;
    dc1394framerate_t framerate;
    dc1394video_mode_t video_mode = 0;
    dc1394color_coding_t coding;
    unsigned int width, height;
    dc1394video_frame_t *frame;
    dc1394_t * d;
    dc1394camera_list_t * list;

    dc1394error_t err;

    d = dc1394_new ();
    if (!d)
        return 1;
    err=dc1394_camera_enumerate (d, &list);
    DC1394_ERR_RTN(err,"Failed to enumerate cameras");

    if (list->num == 0) {
        dc1394_log_error("No cameras found");
        return 1;
    }

    camera = dc1394_camera_new (d, list->ids[0].guid);
    if (!camera) {
        dc1394_log_error("Failed to initialize camera with guid %"PRIx64, list->ids[0].guid);
        return 1;
    }
    dc1394_camera_free_list (list);

    printf("Using camera with GUID %"PRIx64"\n", camera->guid);

    /*-----------------------------------------------------------------------
     *  get the best video mode and highest framerate. This can be skipped
     *  if you already know which mode/framerate you want...
     *-----------------------------------------------------------------------*/
    // get video modes:
    err=dc1394_video_get_supported_modes(camera,&video_modes);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Can't get video modes");

    video_mode = DC1394_VIDEO_MODE_FORMAT7_0;

    err=dc1394_get_color_coding_from_video_mode(camera, video_mode,&coding);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not get color coding");

    framerate=DC1394_FRAMERATE_15;

    /*-----------------------------------------------------------------------
     *  setup capture
     *-----------------------------------------------------------------------*/

    err=dc1394_video_set_iso_speed(camera, DC1394_ISO_SPEED_400);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set iso speed");

    err=dc1394_video_set_mode(camera, video_mode);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set video mode");

    err=dc1394_video_set_framerate(camera, framerate);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not set framerate");

    err=dc1394_capture_setup(camera,4, DC1394_CAPTURE_FLAGS_DEFAULT);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not setup camera-\nmake sure that the video mode and framerate are\nsupported by your camera");
    
    dc1394_get_image_size_from_video_mode(camera, video_mode, &width, &height);
    rgb = (uint8_t*) malloc(width*height);

    err=dc1394_feature_set_value(camera, DC1394_FEATURE_SHUTTER, 240);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Couldn't set shutter");


    /*-----------------------------------------------------------------------
     *  report camera's features
     *-----------------------------------------------------------------------*/
    err=dc1394_feature_get_all(camera,&features);
    if (err!=DC1394_SUCCESS) {
        dc1394_log_warning("Could not get feature set");
    }
    else {
        dc1394_feature_print_all(&features, stdout);
    }

    /*-----------------------------------------------------------------------
     *  have the camera start sending us data
     *-----------------------------------------------------------------------*/
    err=dc1394_video_set_transmission(camera, DC1394_ON);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not start camera iso transmission");

    int j = 0;
    for (j = 0; j < 20; ++j) {
        /*-----------------------------------------------------------------------
         *  capture one frame
         *-----------------------------------------------------------------------*/
        err=dc1394_capture_dequeue(camera, DC1394_CAPTURE_POLICY_WAIT, &frame);
        DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not capture a frame");

        /*
         * Debayer stuff
         * */
        err=dc1394_bayer_decoding_8bit(frame->image, rgb, width, height, DC1394_COLOR_FILTER_RGGB, DC1394_BAYER_METHOD_BILINEAR);
        DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Debayer broke");

        /*-----------------------------------------------------------------------
         *  save image as 'Image.pgm'
         *-----------------------------------------------------------------------*/
        char image_file_name[50];
        sprintf(image_file_name, "image_%d.ppm", j);

        imagefile=fopen(image_file_name, "wb");

        if( imagefile == NULL) {
            char errmsg[50];
            sprintf(errmsg, "Can't create '%s'", image_file_name);
            perror(errmsg);
            cleanup_and_exit(camera);
        }

        fprintf(imagefile,"P6\n%u %u 255\n", width, height);
        fwrite(rgb, 1, height*width, imagefile);
        fclose(imagefile);
        printf("wrote: %s\n", image_file_name);

        err=dc1394_capture_enqueue(camera, frame);
        DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not capture a frame");

        if (j == 5) {
            err=dc1394_feature_set_value(camera, DC1394_FEATURE_SHUTTER, 1);
            DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Couldn't set shutter");
        }

    }


    /*-----------------------------------------------------------------------
     *  stop data transmission
     *-----------------------------------------------------------------------*/
    err=dc1394_video_set_transmission(camera,DC1394_OFF);
    DC1394_ERR_CLN_RTN(err,cleanup_and_exit(camera),"Could not stop the camera");


    /*-----------------------------------------------------------------------
     *  close camera
     *-----------------------------------------------------------------------*/
    dc1394_video_set_transmission(camera, DC1394_OFF);
    dc1394_capture_stop(camera);
    dc1394_camera_free(camera);
    dc1394_free (d);
    free(rgb);

    return 0;
}
