// This file describes a POV-Ray "scene" for the two 3D plots in the manuscript.
// Show/hide the "dependent" and "independent" polygons here:
// To specify output resolution enter "+W2250 +H1800" in the box right of the resolution dropdown.
#declare show_dependent = false;
#declare show_independent = true;

#include "colors.inc"
#include "textures.inc"
background{<1,1,1,0>}

global_settings
{
  max_trace_level 10
}

camera{
    perspective
    location  <4,6,-4>  
    angle 11
    look_at   <0.48,-0.05,0.48>
}

light_source {<3,4,-4> rgb <1, 1, 1>*0.6 shadowless}
light_source {<4,3,-2> rgb <1, 1, 1>*0.6 shadowless}



#declare MIN_X = 25;
#declare MAX_X = 60;
#declare STEP_X = 5;
#declare MIN_Y = 0;
#declare MAX_Y = 25;
#declare MIN_Z = 1.8;
#declare MAX_Z = 3.1;
#declare STEP_Z = 0.1;

#local SIZE_X = MAX_X - MIN_X;
#local SIZE_Z = MAX_Z - MIN_Z;
#local SHIFT = <-MIN_X, -MIN_Y, -MIN_Z>;
#local SCALE = <1/(MAX_X-MIN_X), 1/(MAX_Y-MIN_Y), 1/(MAX_Z-MIN_Z) >;
                 


// coordinate system
union {
    cylinder{<0,0,0>, <1,0,0>, 0.0025 pigment{Blue}}
    // A vertical axis:
    // cylinder{<0,0,0>, <0,0.5,0>, 0.0025 pigment{Black}}
    cylinder{<0,0,0>, <0,0,1>, 0.0025 pigment{Green}}
    #for (i, MIN_X, MAX_X, STEP_X)   
        #local pos = (i - MIN_X) / SIZE_X;
        cylinder{<pos,0,0>, <pos,0,-0.02>, 0.002 pigment{Blue}}   
        text{internal 2 str(i, 3, 0) 0 0 pigment{Blue} rotate<90,0,0> translate<-0.8,0,0> scale 0.05 translate<pos, 0, -0.07>}
    #end
    #for (i, MIN_Z, MAX_Z, STEP_Z)   
        #local pos = (i - MIN_Z) / SIZE_Z;
        cylinder{<0,0,pos>, <-0.02,0,pos>, 0.002 pigment{Green}}   
        text{internal 2 str(i, 3, 1) 0 0 pigment{Green} rotate<90,0,0> translate<0,0,-0.25> scale 0.05 translate<-0.1, 0, pos>}
    #end
    
    text{internal 2 "independent" 0 0 pigment{Blue} rotate<90,0,0> translate<0,0,-0.4> scale 0.05 translate<0.35, 0, -0.12>}
    text{internal 2 "dependent" 0 0 pigment{Green} rotate<90,-90,0> translate<0,0,-0.4> scale 0.05 translate<-0.15, 0, 0.5>}
}

        
  
// plot content
union{
    // the left/right.inc files are written by the Jupyter notebook!
    #if (show_dependent)
        #include "dependent.inc"
    #end
    #if (show_independent)
        #include "independent.inc"
    #end
    translate <-MIN_X, -MIN_Y, -MIN_Z>
    scale <1/(MAX_X-MIN_X), 1/(MAX_Y-MIN_Y), 1/(MAX_Z-MIN_Z) >    
}
                                                        
                                                        
                                                        
                                                        
                                                        
         