#include "directed_driver.h"

int main(int argc, char **argv)
{
    static const bcd_problem acvrp = {
        .key = "acvrp",
        .prog = "bc_acvrp",
        .summary = "Branch-and-cut for the asymmetric CVRP with RCC separation.",
        .model_line =
            "Directed two-index model of Fischetti, Toth and Vigo (1994): binary arcs on\n"
            "every ordered pair, degree one at every customer, exactly K arcs out of the\n"
            "depot and K back in, and the asymmetric matrix used as written.",
        .open_routes = false,
        .distance_flag = false,
    };
    return bcd_main(argc, argv, &acvrp);
}
