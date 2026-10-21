#ifndef TW_SOLOMON_H
#define TW_SOLOMON_H

#include <stdint.h>

#define TW_NAME_MAX 128

typedef enum {
    TW_DISTANCE_TRUNC1 = 0,
    TW_DISTANCE_EXACT  = 1
} tw_distance;

_Bool tw_distance_parse(const char *text, tw_distance *out);
const char *tw_distance_name(tw_distance convention);

typedef struct {
    char     name[TW_NAME_MAX];
    int64_t  n_customers;
    int64_t  capacity;
    int64_t  available_vehicles;
    int64_t *demand;
    double  *coords;
    double  *ready;
    double  *due;
    double  *service;
    double  *travel;
    tw_distance convention;
} tw_instance;

void tw_instance_load(const char *path, tw_distance convention, tw_instance *out);
void tw_instance_free(tw_instance *out);

typedef struct {
    int64_t  n_model_nodes;
    int64_t  n_arcs;
    int32_t *tail;
    int32_t *head;
    double  *cost;
    double  *time;
    double  *big_m;
    double  *lower;
    double  *upper;
    double  *service;
    int64_t  n_arcs_before;
    int64_t  removed_time_window;
    int64_t  removed_pair_capacity;
} tw_arcs;

void tw_preprocess_arcs(const tw_instance *inst, tw_arcs *out);
void tw_arcs_free(tw_arcs *arcs);

void tw_check_customer_degrees(const tw_arcs *arcs, int64_t n_customers);

#endif
