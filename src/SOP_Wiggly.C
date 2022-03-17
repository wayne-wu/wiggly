


#include <UT/UT_DSOVersion.h>
#include <UT/UT_Math.h>
#include <UT/UT_Interrupt.h>
#include <GU/GU_Detail.h>
#include <GU/GU_PrimPoly.h>
#include <CH/CH_LocalVariable.h>
#include <PRM/PRM_Include.h>
#include <PRM/PRM_TemplateBuilder.h>
#include <OP/OP_Operator.h>
#include <OP/OP_OperatorTable.h>
#include <OP/OP_AutoLockInputs.h>


#include <limits.h>
#include "SOP_Wiggly.h"
#include "SOP_Wiggly.proto.h"

using namespace HDK_Sample;


const UT_StringHolder SOP_Wiggly::theSOPTypeName("hdk_wiggly");

void
newSopOperator(OP_OperatorTable *table)
{
    table->addOperator(
	    new OP_Operator("wiggly",			// Internal name
			    "Wiggly",			// UI name
			     SOP_Wiggly::myConstructor,	// How to build the SOP
			     SOP_Wiggly::buildTemplates(),	// My parameters
			     2,				// Min # of sources
			     2,				// Max # of sources
					 nullptr,
			     OP_FLAG_GENERATOR)		// Flag it as generator
	    );
}

static const char* theDsFile = R"THEDSFILE(
{
        name        parameters
        parm {
            name    "divs"      // Internal parameter name
            label   "Divisions" // Descriptive parameter name for user interface
            type    integer
            default { "5" }     // Default for this parameter on new nodes
            range   { 2! 50 }   // The value is prevented from going below 2 at all.
                                // The UI slider goes up to 50, but the value can go higher.
            export  all         // This makes the parameter show up in the toolbox
                                // above the viewport when it's in the node's state.
       }
}
)THEDSFILE";

PRM_Template*
SOP_Wiggly::buildTemplates()
{
	static PRM_TemplateBuilder templ("SOP_Wiggly.C", theDsFile);
	return templ.templates();
}

class SOP_WigglyVerb : public SOP_NodeVerb
{
public:
	SOP_WigglyVerb() {}
	virtual ~SOP_WigglyVerb() {}

	virtual SOP_NodeParms* allocParms() const { return new SOP_WigglyParms(); }
	virtual UT_StringHolder name() const { return SOP_Wiggly::theSOPTypeName; }

	virtual SOP_NodeVerb::CookMode cookMode(const SOP_NodeParms* parms) const { return SOP_NodeVerb::COOK_GENERIC; }

	virtual void cook(const SOP_NodeVerb::CookParms& cookparms) const;

	static const SOP_NodeVerb::Register<SOP_WigglyVerb> theVerb;
};

const SOP_NodeVerb::Register<SOP_WigglyVerb> SOP_WigglyVerb::theVerb;

const SOP_NodeVerb* 
SOP_Wiggly::cookVerb() const
{
	return SOP_WigglyVerb::theVerb.get();
}


void
SOP_WigglyVerb::cook(const SOP_NodeVerb::CookParms& cookparms) const
{
		// fpreal		 now = context.getTime();
		auto&& sopparms = cookparms.parms<SOP_WigglyParms>();
		GU_Detail* gdp = cookparms.gdh().gdpNC();

		// const GU_Detail * bgdp = inputGeo(1);  // constraints
		// const GA_Attribute * bh = bgdp->findAttribute(GA_ATTRIB_POINT, "frame");

		// Loop through all points and modify the value
		GA_Offset ptoff;
		GA_FOR_ALL_PTOFF(gdp, ptoff)
		{
			UT_Vector3 p = gdp->getPos3(ptoff);

			p = -p;

			gdp->setPos3(ptoff, p);
		}
	
		gdp->getP()->bumpDataId();
}

