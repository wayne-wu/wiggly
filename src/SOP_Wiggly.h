#ifndef __SOP_WIGGLY_h
#define __SOP_WIGGLY_h

#include <SOP/SOP_Node.h>
#include <SOP/SOP_NodeVerb.h>
#include <UT/UT_StringHolder.h>

namespace HDK_Wiggly {

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
      mySopFlags.setNeedGuide1(true);
    }
    ~SOP_Wiggly() override {}

    // delegates to the verb
    OP_ERROR cookMySop(OP_Context& context) override
    {
      flags().setTimeDep(true);  //NOTE: Is this the right way to do it?
      return cookMyselfAsVerb(context);
    }

    OP_ERROR cookMyGuide1(OP_Context& context) override
    {
      OP_AutoLockInputs inputs(this);
      if (inputs.lock(context) >= UT_ERROR_ABORT)
        return error();
      myGuide1->clearAndDestroy();
      myGuide1->copy(*inputGeo(1, context));
      return error();
    }

    const char* inputLabel(unsigned idx) const override
    {
      switch (idx)
      {
      case 0:   return "Tetrahedral Rest Mesh";
      case 1:   return "Constraints";
      default:  return "Invalid Source";
      }
    }
};
} // End HDK_Sample namespace

#endif