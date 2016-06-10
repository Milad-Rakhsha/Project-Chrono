// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Milad Rakhsha
// =============================================================================
//
// This demo shows how HEEDS-MDO can be linked to CHRONO.
// =============================================================================

#include "chrono/core/ChFileutils.h"
#include "chrono/ChConfig.h"

#include "thirdparty/rapidjson/document.h"
#include "thirdparty/rapidjson/filereadstream.h"

#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/physics/ChLoaderUV.h"

#include <iostream>
#include <math.h>
#include <sstream>
#include <fstream>
#include <string>
#include <algorithm>
#include <functional>
#include <stdio.h>

using namespace chrono;
using namespace std;
using namespace rapidjson;

double Input1, Input2;

void SetParamFromJSON(const std::string& filename, double& Input1,double& Input2);

int main(int argc, char* argv[]) {
    SetParamFromJSON(argv[1], Input1, Input2);
    std::ofstream output_file;
    output_file.open(argv[2]);
    double m_output = Input1+ (Input2 * Input2) / 2;
    output_file << m_output << "\n";
    output_file.close();
    cout <<"The Output is: " << m_output << "\n\n\n";

    return 0;
}
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================
//======================================================================================================================================

void SetParamFromJSON(const std::string& filename, double& Input1,double& Input2) {
    // -------------------------------------------
    // Open and parse the input file
    // -------------------------------------------
    FILE* fp = fopen(filename.c_str(), "r");

    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    fclose(fp);

    Document d;
    d.ParseStream(is);

    // Read top-level data
    assert(d.HasMember("Chrono Inputs"));
    Input1 = d["Chrono Inputs"]["x"].GetDouble();
    Input2 = d["Chrono Inputs"]["y"].GetDouble();
    //GetLog() << "Loaded JSON: " << filename.c_str() << "\n";
}
