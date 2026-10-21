#include "directed_driver.h"

int main(int argc, char **argv)
{
    static const bcd_problem ovrp = {
        .key = "ovrp",
        .prog = "bc_ovrp",
        .summary = "Branch-and-cut for the open VRP with RCC separation.",
        .model_line =
            "Virtual-sink model of Letchford, Lysgaard and Eglese (2007): arcs run from the\n"
            "depot to a customer, between customers, and from a customer to a zero-cost\n"
            "sink, with exactly K leaving the depot and K entering the sink.",
        .open_routes = true,
        .distance_flag = true,
    };
    return bcd_main(argc, argv, &ovrp);
}
