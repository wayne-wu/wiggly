


#include <UT/UT_DSOVersion.h>
//#include <RE/RE_EGLServer.h>


#include <UT/UT_Math.h>
#include <UT/UT_Interrupt.h>
#include <GU/GU_Detail.h>
#include <GU/GU_PrimPoly.h>
#include <CH/CH_LocalVariable.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_SpareData.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>


#include <limits.h>
#include "SOP_Wiggly.h"

using namespace HDK_Sample;

//
// Help is stored in a "wiki" style text file. 
//
// See the sample_install.sh file for an example.
//
// NOTE : Follow this tutorial if you have any problems setting up your visual studio 2008 for Houdini 
//  http://www.apileofgrains.nl/setting-up-the-hdk-for-houdini-12-with-visual-studio-2008/


///
/// newSopOperator is the hook that Houdini grabs from this dll
/// and invokes to register the SOP.  In this case we add ourselves
/// to the specified operator table.
///
void
newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(
	    new OP_Operator("wiggly",			// Internal name
			    "Wiggly",			// UI name
			     SOP_Wiggly::myConstructor,	// How to build the SOP
			     SOP_Wiggly::myTemplateList,	// My parameters
			     1,				// Min # of sources
			     1,				// Max # of sources
					 nullptr,
			     OP_FLAG_GENERATOR)		// Flag it as generator
	    );
}


PRM_Template
SOP_Wiggly::myTemplateList[] = {
    PRM_Template()
};

OP_Node *
SOP_Wiggly::myConstructor(OP_Network *net, const char *name, OP_Operator *op)
{
	return new SOP_Wiggly(net, name, op);
}

SOP_Wiggly::SOP_Wiggly(OP_Network *net, const char *name, OP_Operator *op)
	: SOP_Node(net, name, op)
{
}

SOP_Wiggly::~SOP_Wiggly() {}

unsigned
SOP_Wiggly::disableParms()
{
    return 0;
}

OP_ERROR
SOP_Wiggly::cookMySop(OP_Context &context)
{
		fpreal		 now = context.getTime();

    return error();
}

