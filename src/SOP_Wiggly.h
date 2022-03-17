#ifndef __SOP_WIGGLY_h
#define __SOP_WIGGLY_h

#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>
#include <UT/UT_StringHolder.h>

namespace HDK_Sample {

  class SOP_Wiggly : public SOP_Node
  {
  public:
    static OP_Node* myConstructor(OP_Network* net, const char* name, OP_Operator* op)
    {
      return new SOP_Wiggly(net, name, op);
    }

    /// Stores the description of the interface of the SOP in Houdini.
    /// Each parm template refers to a parameter.
    static PRM_Template* buildTemplates();

    static const UT_StringHolder theSOPTypeName;

    const SOP_NodeVerb* cookVerb() const override;

  protected:
    SOP_Wiggly(OP_Network* net, const char* name, OP_Operator* op)
      : SOP_Node(net, name, op) 
    {
      mySopFlags.setManagesDataIDs(true);
    }
    ~SOP_Wiggly() override {}

    // delegates to the verb
    OP_ERROR		 cookMySop(OP_Context& context)
    {
      return cookMyselfAsVerb(context);
    }
};
} // End HDK_Sample namespace

#endif