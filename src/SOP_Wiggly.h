#ifndef __SOP_WIGGLY_h
#define __SOP_WIGGLY_h

//#include <GEO/GEO_Point.h>
//
#include <SOP/SOP_Node.h>

namespace HDK_Sample {

  class SOP_Wiggly : public SOP_Node
  {
  public:
    static OP_Node* myConstructor(OP_Network*, const char*,
      OP_Operator*);

    SOP_Wiggly(OP_Network* net, const char* name, OP_Operator* op);
    virtual ~SOP_Wiggly();

    /// Stores the description of the interface of the SOP in Houdini.
    /// Each parm template refers to a parameter.
    static PRM_Template		 myTemplateList[];

    /// This optional data stores the list of local variables.
    static CH_LocalVariable	 myVariables[];

  protected:

    /// Disable parameters according to other parameters.
    virtual unsigned		 disableParms();


    /// cookMySop does the actual work of the SOP computing, in this
    /// case, a LSYSTEM
    virtual OP_ERROR		 cookMySop(OP_Context& context);

  private:
    /// The following list of accessors simplify evaluating the parameters
    /// of the SOP.
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    /// Member variables are stored in the actual SOP, not with the geometry
    /// In this case these are just used to transfer data to the local 
    /// variable callback.
    /// Another use for local data is a cache to store expensive calculations.

	  // NOTE : You can declare local variables here like this  
    // int		myCurrPoint;
    // int		myTotalPoints;
};
} // End HDK_Sample namespace

#endif