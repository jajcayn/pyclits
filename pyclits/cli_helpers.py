#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def process_CLI_arguments(arguments, separator=[",", "'", "/", "|"]):
    processed_arguments = []
    neu_set = []
    for item in arguments:
        if item in separator:
            processed_arguments.append(neu_set)
            neu_set = []
        else:
            neu_set.append(int(item))

    processed_arguments.append(neu_set)
    return processed_arguments
